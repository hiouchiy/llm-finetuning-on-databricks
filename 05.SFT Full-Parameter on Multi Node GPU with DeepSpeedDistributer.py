# Databricks notebook source
# MAGIC %md
# MAGIC ## Databricks ML Runtime: 17.3 ML LTS
# MAGIC ## Instance Type: Standard_NC24ads_A100_v4 [A100x1] x 4 nodes

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Zero2-Offload無し

# COMMAND ----------

import os
import sys
import json
import inspect
from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor

# Faster HF downloads (optional)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# =========================
# クラスタ構成
# numGpus は「1ノードあたりのGPU枚数」
# =========================
NNODES = int(os.environ.get("NNODES", "4"))
NUM_GPUS_PER_NODE = int(os.environ.get("NUM_GPUS_PER_NODE", "1"))

# ========================================
# Databricks認証情報を取得して環境変数に設定
# ========================================
def get_databricks_credentials():
    """ノートブック環境からDatabricks認証情報を取得"""
    try:
        from dbruntime.databricks_repl_context import get_context
        context = get_context()
        host = context.apiUrl
        token = context.apiToken
        return host, token
    except Exception as e:
        print(f"Warning: Could not get credentials automatically: {e}")
        return None, None

DATABRICKS_HOST, DATABRICKS_TOKEN = get_databricks_credentials()

if DATABRICKS_HOST and DATABRICKS_TOKEN:
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
    # spark.conf.set("spark.executorEnv.DATABRICKS_HOST", DATABRICKS_HOST)
    # spark.conf.set("spark.executorEnv.DATABRICKS_TOKEN", DATABRICKS_TOKEN)
    # print(f"✅ Set DATABRICKS_HOST: {DATABRICKS_HOST}")
else:
    print("⚠️ Could not retrieve Databricks credentials")

# ========================================
# 標準出力を画面とファイルに同時出力するクラス
# ========================================
class TeeLogger:
    """標準出力を画面とファイルに同時に書き込む"""
    def __init__(self, log_file, mode="a"):
        self.terminal = sys.stdout
        self.log = open(log_file, mode, buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ========================================
# トレーニング関数（ZeRO-2版）
# ========================================
def train_nemotron_fullft_zero2(host: str, token: str, exp_name: str, run_id: str):
    import os
    import sys
    import time
    import torch
    import torch.distributed as dist
    import mlflow
    from mlflow.tracking import MlflowClient
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from trl import SFTTrainer, SFTConfig

    # ---- ノイズ抑制（TF/XLA系の警告が出る環境向け）----
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # ---- NCCL / distributed env (init 前) ----
    # 旧NCCL_* がdeprecated警告になる環境があるので TORCH_NCCL_* に寄せる
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")
    os.environ["NCCL_DEBUG_SUBSYS"] = os.environ.get("NCCL_DEBUG_SUBSYS", "INIT,NET")
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = os.environ.get("TORCH_DISTRIBUTED_DEBUG", "DETAIL")

    # ネットワークIFは環境に合わせて。迷ったら eth0 を維持
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")

    # IBが無い/不安定なら 1 のまま
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    # deprecated になりがちなキーを削除して TORCH_NCCL_* を使う
    for k in ["NCCL_ASYNC_ERROR_HANDLING", "NCCL_BLOCKING_WAIT"]:
        if k in os.environ:
            del os.environ[k]
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    # NCCL_P2P_DISABLE は外す（＝P2P有効）
    if "NCCL_P2P_DISABLE" in os.environ:
        del os.environ["NCCL_P2P_DISABLE"]

    # ---- ranks / device ----
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    _ = torch.cuda.current_device()  # CUDAコンテキスト確立の意図

    # init_process_group（PyTorchが対応していれば device_id も渡す）
    if not dist.is_initialized():
        init_kwargs = {"backend": "nccl", "init_method": "env://"}
        sig = inspect.signature(dist.init_process_group)
        if "device_id" in sig.parameters:
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**init_kwargs)

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_global0 = (global_rank == 0)

    client = None
    if is_global0:
        mlflow.set_tracking_uri("databricks")
        os.environ["DATABRICKS_HOST"] = host
        os.environ["DATABRICKS_TOKEN"] = token
        os.environ["MLFLOW_EXPERIMENT_NAME"] = exp_name
        # 念のため（無ければ失敗させて気付けるように）
        assert os.environ.get("DATABRICKS_HOST")
        assert os.environ.get("DATABRICKS_TOKEN")
        exp = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if exp:
            mlflow.set_experiment(exp)

        mlflow.end_run()  # 念のため、勝手に開始されている active run を落とす
        client = MlflowClient()

    # mlflow_run_id = os.environ.get("MLFLOW_RUN_ID", run_id)
    mlflow_run_id = run_id

    # rank0のみログ
    log_file_path = "/tmp/training_output.log"
    tee = None
    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    try:
        if is_global0:
            tee = TeeLogger(log_file_path, mode="w")
            sys.stdout = tee
            sys.stderr = tee
            print(f"📝 Logging all output to {log_file_path}")
            print(f"🧭 global_rank={global_rank} world_size={world_size} local_rank={local_rank}")
            print(f"🖥️ torch.cuda.device_count()={torch.cuda.device_count()}")
            sys.stdout.flush()

        # ====== 進捗バー/ログ抑制（非rank0は静かに）======
        if not is_global0:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            try:
                from datasets.utils.logging import disable_progress_bar
                disable_progress_bar()
            except Exception:
                pass

        # ========================================
        # MLflowコールバック（global rank 0のみ）
        # ========================================
        class MLflowLoggingCallback(TrainerCallback):
            def __init__(self, run_id, log_file, is_global0: bool, client):
                self.run_id = run_id
                self.log_file = log_file
                self.is_global0 = is_global0
                self.last_log_time = None
                self.upload_threads = []
                self.client = client
            
            def on_train_begin(self, args, state, control, **kwargs):
                """トレーニング開始時に呼ばれる"""
                if self.is_global0 and self.run_id:
                    self.client.log_artifact(self.run_id, self.log_file, artifact_path="logs")
                    print(f"✅ Training loop started! Total steps: {state.max_steps}")  # ← 追加

            def on_log(self, args, state, control, logs=None, **kwargs):
                if self.is_global0 and logs and self.run_id:
                    try:
                        current_time = time.time()

                        if self.last_log_time is not None:
                            elapsed = current_time - self.last_log_time
                            time_per_step = elapsed / max(int(getattr(args, "logging_steps", 1)), 1)
                            logs["time_per_step"] = time_per_step

                        self.last_log_time = current_time

                        # with mlflow.start_run(run_id=self.run_id):
                        for key, value in logs.items():
                            if isinstance(value, (int, float)):
                                # mlflow.log_metric(key, value, step=state.global_step)
                                self.client.log_metric(self.run_id, key, value, step=state.global_step)

                        if state.global_step % 100 == 0:
                            # mlflow.log_artifact(self.log_file, artifact_path="logs")
                            self.client.log_artifact(self.run_id, self.log_file, artifact_path="logs")
                            print(f"📤 Uploaded training log (step {state.global_step})")
                            sys.stdout.flush()

                    except Exception as e:
                        print(f"Warning: MLflow logging failed at step {state.global_step}: {e}")
                        sys.stdout.flush()

            def on_save(self, args, state, control, **kwargs):
                if self.is_global0 and self.run_id:
                    import threading
                    import os

                    checkpoint_folder = f"checkpoint-{state.global_step}"
                    checkpoint_path = os.path.join(args.output_dir, checkpoint_folder)
                    if not os.path.exists(checkpoint_path):
                        return

                    def upload_checkpoint():
                        try:
                            # with mlflow.start_run(run_id=self.run_id):
                            print(f"📤 Uploading {checkpoint_folder} to MLflow (async)...")
                            # mlflow.log_artifacts(
                            #     checkpoint_path,
                            #     artifact_path=f"checkpoints/{checkpoint_folder}",
                            # )
                            self.client.log_artifact(self.run_id, checkpoint_path, artifact_path=f"checkpoints/{checkpoint_folder}")
                            print(f"✅ {checkpoint_folder} uploaded to MLflow")
                            sys.stdout.flush()
                        except Exception as e:
                            print(f"❌ Checkpoint upload failed for {checkpoint_folder}: {e}")
                            sys.stdout.flush()

                    thread = threading.Thread(target=upload_checkpoint, daemon=False)
                    thread.start()
                    self.upload_threads.append(thread)
                    self.upload_threads = [t for t in self.upload_threads if t.is_alive()]

            def on_train_end(self, args, state, control, **kwargs):
                if self.is_global0:
                    if self.upload_threads:
                        print(f"⏳ Waiting for {len(self.upload_threads)} uploads to complete...")
                        for thread in self.upload_threads:
                            thread.join()
                        print("✅ All checkpoint uploads completed")

                    if self.run_id:
                        try:
                            # with mlflow.start_run(run_id=self.run_id):
                                # mlflow.log_artifact(self.log_file, artifact_path="logs")
                            self.client.log_artifact(self.run_id, self.log_file, artifact_path="logs")
                            print("✅ Final training log uploaded to MLflow")
                            sys.stdout.flush()
                        except Exception as e:
                            print(f"Warning: Final log upload failed: {e}")
                            sys.stdout.flush()

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if self.is_global0 and metrics and self.run_id:
                    try:
                        # with mlflow.start_run(run_id=self.run_id):
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                # mlflow.log_metric(f"eval_{key}", value, step=state.global_step)
                                self.client.log_metric(self.run_id, f"eval_{key}", value, step=state.global_step)
                    except Exception as e:
                        print(f"Warning: MLflow eval logging failed: {e}")
                        sys.stdout.flush()

        MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
        DATASET_ID = "bbz662bbz/databricks-dolly-15k-ja-gozaru"

        # trust_remote_code を使うなら本来は revision pin 推奨（ここでは任意）
        # MODEL_REVISION = os.environ.get("MODEL_REVISION")  # 例: コミットSHA
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, revision=MODEL_REVISION)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def build_user_text(ex):
            inst = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            return f"{inst}\n\n[入力]\n{inp}" if inp else inst

        def to_text(ex):
            messages = [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": build_user_text(ex)},
                {"role": "assistant", "content": (ex.get("output") or "").strip()},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}

        if is_global0:
            print("📥 Loading dataset...")
            sys.stdout.flush()

        ds = load_dataset(DATASET_ID, split="train")
        ds = ds.map(to_text, remove_columns=ds.column_names)

        if is_global0:
            print("📦 Loading model...")
            sys.stdout.flush()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = model.to(local_rank)
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        output_dir = "/local_disk0/nemotron_nano_9b_gozaru_fullft_zero2"

        # ===============================
        # DeepSpeed ZeRO-2 config
        # ===============================
        ds_config = {
            "wall_clock_breakdown": True,
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_scatter": True,
            },
        }
        ds_config_path = "/tmp/ds_zero2_no_offload.json"
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f)

        args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            bf16=True,
            deepspeed=ds_config_path,
            optim="adamw_torch_fused",
            report_to=[],
            max_length=2048,
            packing=False,
            disable_tqdm=(not is_global0),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        callbacks = []
        if mlflow_run_id:
            callbacks.append(
                MLflowLoggingCallback(run_id=mlflow_run_id, log_file=log_file_path, is_global0=is_global0, client=client)
            )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=ds,
            args=args,
            callbacks=callbacks,
        )

        if is_global0:
            print("🚀 Starting training...")
            sys.stdout.flush()

        train_result = trainer.train()

        # global rank 0のみ保存 & MLflowへ成果物登録（driverではなくworker側で！）
        if is_global0:
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print("✅ Training done (ZeRO-2)")
            print("model_dir:", output_dir)
            sys.stdout.flush()

            # worker上の /local_disk0 を、その場でMLflowへアップロード
            if mlflow_run_id:
                try:
                    with mlflow.start_run(run_id=mlflow_run_id):
                        mlflow.log_artifacts(output_dir, artifact_path="model")
                        mlflow.log_artifact(log_file_path, artifact_path="logs")
                        print("✅ Uploaded model + final log to MLflow (from worker rank0)")
                        sys.stdout.flush()
                except Exception as e:
                    print(f"Warning: MLflow model/log upload failed: {e}")
                    sys.stdout.flush()

            return {"model_dir": output_dir, "log_file": log_file_path, "metrics": train_result.metrics}

        return None

    finally:
        print("Finished", flush=True)
        # stdout/stderr restore
        try:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            if tee is not None:
                tee.close()
        except Exception:
            pass

# ========================================
# メインセル：MLflow Run作成 → 学習実行
# ========================================
import mlflow

username = spark.sql("SELECT current_user()").collect()[0][0]
MLFLOW_EXPERIMENT_NAME = f"/Workspace/Users/{username}/nemotron_v2_nano_gozaru_fullft_multi_node"

if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
    mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "TRUE"
os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT_NAME

with mlflow.start_run(run_name="nemotron_nano_9b_gozaru_fullft_sft_zero2") as run:
    mlflow.set_tag("base_model", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    mlflow.set_tag("dataset", "bbz662bbz/databricks-dolly-15k-ja-gozaru")
    mlflow.set_tag("task", "SFT full-parameter finetuning (DeepSpeed ZeRO-2)")

    mlflow.log_params(
        {
            "epochs": 1,
            "per_device_train_batch_size": 1,
            "grad_accum": 8,
            "lr": 2e-4,
            "nnodes": NNODES,
            "gpus_per_node": NUM_GPUS_PER_NODE,
        }
    )

    os.environ["MLFLOW_RUN_ID"] = run.info.run_id

    distributor = DeepspeedTorchDistributor(
        numGpus=NUM_GPUS_PER_NODE,  # per-node GPU count
        nnodes=NNODES,
        localMode=False,
        useGpu=True,
        deepspeedConfig=None,
    )

    result = distributor.run(
        train_nemotron_fullft_zero2, 
        host=DATABRICKS_HOST, 
        token=DATABRICKS_TOKEN, 
        exp_name=MLFLOW_EXPERIMENT_NAME, 
        run_id=run.info.run_id)

    # result["model_dir"] は worker の /local_disk0 なので driver からは触らない（アップロードは worker 側で実施済み）

print("✅ All done!")

# COMMAND ----------

# MAGIC %md
# MAGIC # DeepSpeed Distributorを使用したLLMファインチューニング解説
# MAGIC
# MAGIC ## 📖 はじめに
# MAGIC
# MAGIC このコードは、**大規模言語モデル（LLM）を複数のGPUを使って効率的にファインチューニング（追加学習）する**ためのものです。
# MAGIC
# MAGIC ### このコードで行うこと
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │  NVIDIA Nemotron-Nano-9B（90億パラメータのLLM）             │
# MAGIC │           ↓                                                 │
# MAGIC │  日本語データセット（ござる口調）で追加学習                 │
# MAGIC │           ↓                                                 │
# MAGIC │  日本語で「〜でござる」と答えるモデルに！                   │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔧 コードの全体構成
# MAGIC
# MAGIC ```
# MAGIC ┌────────────────────────────────────────────────────────────────┐
# MAGIC │                        メインセル                              │
# MAGIC │  ┌──────────────────────────────────────────────────────────┐ │
# MAGIC │  │ 1. MLflow実験の設定                                      │ │
# MAGIC │  │ 2. DeepspeedTorchDistributor の起動                      │ │
# MAGIC │  │ 3. 学習関数の分散実行                                    │ │
# MAGIC │  └──────────────────────────────────────────────────────────┘ │
# MAGIC │                            ↓                                   │
# MAGIC │  ┌──────────────────────────────────────────────────────────┐ │
# MAGIC │  │           train_nemotron_fullft_zero2 関数               │ │
# MAGIC │  │  ┌────────────────────────────────────────────────────┐  │ │
# MAGIC │  │  │ • 分散学習の初期化                                 │  │ │
# MAGIC │  │  │ • モデル・データセットの読み込み                   │  │ │
# MAGIC │  │  │ • SFTTrainerによる学習実行                         │  │ │
# MAGIC │  │  │ • MLflowへのログ・モデル保存                       │  │ │
# MAGIC │  │  └────────────────────────────────────────────────────┘  │ │
# MAGIC │  └──────────────────────────────────────────────────────────┘ │
# MAGIC └────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 セクション別解説
# MAGIC
# MAGIC ### 1. インポートと環境設定
# MAGIC
# MAGIC ```python
# MAGIC import os
# MAGIC import sys
# MAGIC import json
# MAGIC import inspect
# MAGIC from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor
# MAGIC
# MAGIC # Faster HF downloads (optional)
# MAGIC os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# MAGIC ```
# MAGIC
# MAGIC #### 💡 ポイント解説
# MAGIC
# MAGIC | ライブラリ | 役割 |
# MAGIC |-----------|------|
# MAGIC | `DeepspeedTorchDistributor` | Databricks上で複数GPUに学習を分散させる |
# MAGIC | `HF_HUB_ENABLE_HF_TRANSFER` | Hugging Faceからのダウンロードを高速化 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. クラスタ構成の設定
# MAGIC
# MAGIC ```python
# MAGIC NNODES = int(os.environ.get("NNODES", "4"))
# MAGIC NUM_GPUS_PER_NODE = int(os.environ.get("NUM_GPUS_PER_NODE", "1"))
# MAGIC ```
# MAGIC
# MAGIC #### 💡 分散学習の構成イメージ
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    クラスタ全体                             │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
# MAGIC │  │   Node 1    │  │   Node 2    │  │   Node 3    │  │   Node 4    │
# MAGIC │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │
# MAGIC │  │  │ GPU 0 │  │  │  │ GPU 0 │  │  │  │ GPU 0 │  │  │  │ GPU 0 │  │
# MAGIC │  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │
# MAGIC │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
# MAGIC │                                                             │
# MAGIC │  NNODES = 4（ノード数）                                     │
# MAGIC │  NUM_GPUS_PER_NODE = 1（1ノードあたりのGPU数）              │
# MAGIC │  → 合計 4 GPU で分散学習                                    │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3. Databricks認証情報の取得
# MAGIC
# MAGIC ```python
# MAGIC def get_databricks_credentials():
# MAGIC     """ノートブック環境からDatabricks認証情報を取得"""
# MAGIC     try:
# MAGIC         from dbruntime.databricks_repl_context import get_context
# MAGIC         context = get_context()
# MAGIC         host = context.apiUrl
# MAGIC         token = context.apiToken
# MAGIC         return host, token
# MAGIC     except Exception as e:
# MAGIC         print(f"Warning: Could not get credentials automatically: {e}")
# MAGIC         return None, None
# MAGIC ```
# MAGIC
# MAGIC #### 💡 なぜ認証情報が必要？
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │  Databricks ノートブック（Driver）                          │
# MAGIC │       │                                                     │
# MAGIC │       │ 認証情報を渡す                                      │
# MAGIC │       ↓                                                     │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │  Worker ノード（実際に学習を実行）                   │   │
# MAGIC │  │       │                                              │   │
# MAGIC │  │       │ MLflowにアクセスするために認証が必要         │   │
# MAGIC │  │       ↓                                              │   │
# MAGIC │  │  ┌─────────────────────────────────────────────┐    │   │
# MAGIC │  │  │  MLflow Tracking Server                     │    │   │
# MAGIC │  │  │  （学習ログ・モデルを保存）                 │    │   │
# MAGIC │  │  └─────────────────────────────────────────────┘    │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4. TeeLoggerクラス（ログ出力の二重化）
# MAGIC
# MAGIC ```python
# MAGIC class TeeLogger:
# MAGIC     """標準出力を画面とファイルに同時に書き込む"""
# MAGIC     def __init__(self, log_file, mode="a"):
# MAGIC         self.terminal = sys.stdout
# MAGIC         self.log = open(log_file, mode, buffering=1)
# MAGIC
# MAGIC     def write(self, message):
# MAGIC         self.terminal.write(message)
# MAGIC         self.log.write(message)
# MAGIC         self.log.flush()
# MAGIC ```
# MAGIC
# MAGIC #### 💡 動作イメージ
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    print("Hello!")                          │
# MAGIC │                          │                                  │
# MAGIC │                          ↓                                  │
# MAGIC │                    ┌──────────┐                             │
# MAGIC │                    │TeeLogger │                             │
# MAGIC │                    └──────────┘                             │
# MAGIC │                     ↙        ↘                              │
# MAGIC │           ┌─────────┐      ┌─────────────┐                  │
# MAGIC │           │ 画面表示 │      │ ファイル保存 │                  │
# MAGIC │           │ (stdout)│      │ (.log)      │                  │
# MAGIC │           └─────────┘      └─────────────┘                  │
# MAGIC │                                                             │
# MAGIC │  → 学習の進捗を確認しつつ、後でログを見返せる               │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5. メイン学習関数の構造
# MAGIC
# MAGIC ```python
# MAGIC def train_nemotron_fullft_zero2(host: str, token: str, exp_name: str, run_id: str):
# MAGIC ```
# MAGIC
# MAGIC この関数は**各Workerノードで実行**されます。大きく分けて以下の処理を行います：
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │              train_nemotron_fullft_zero2 関数               │
# MAGIC │                                                             │
# MAGIC │  ┌───────────────────────────────────────────────────────┐ │
# MAGIC │  │ Phase 1: 環境設定・分散学習の初期化                   │ │
# MAGIC │  │   • NCCL設定（GPU間通信の設定）                       │ │
# MAGIC │  │   • プロセスグループの初期化                          │ │
# MAGIC │  │   • rank（自分が何番目のGPUか）の確認                 │ │
# MAGIC │  └───────────────────────────────────────────────────────┘ │
# MAGIC │                          ↓                                  │
# MAGIC │  ┌───────────────────────────────────────────────────────┐ │
# MAGIC │  │ Phase 2: データ・モデルの準備                         │ │
# MAGIC │  │   • トークナイザーの読み込み                          │ │
# MAGIC │  │   • データセットの読み込みと前処理                    │ │
# MAGIC │  │   • モデルの読み込み                                  │ │
# MAGIC │  └───────────────────────────────────────────────────────┘ │
# MAGIC │                          ↓                                  │
# MAGIC │  ┌───────────────────────────────────────────────────────┐ │
# MAGIC │  │ Phase 3: 学習の実行                                   │ │
# MAGIC │  │   • DeepSpeed設定                                     │ │
# MAGIC │  │   • SFTTrainerによる学習                              │ │
# MAGIC │  │   • コールバックでMLflowにログ                        │ │
# MAGIC │  └───────────────────────────────────────────────────────┘ │
# MAGIC │                          ↓                                  │
# MAGIC │  ┌───────────────────────────────────────────────────────┐ │
# MAGIC │  │ Phase 4: 保存・クリーンアップ                         │ │
# MAGIC │  │   • モデルの保存                                      │ │
# MAGIC │  │   • MLflowへのアップロード                            │ │
# MAGIC │  └───────────────────────────────────────────────────────┘ │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 6. 分散学習の初期化部分
# MAGIC
# MAGIC ```python
# MAGIC # ---- ranks / device ----
# MAGIC local_rank = int(os.environ.get("LOCAL_RANK", "0"))
# MAGIC torch.cuda.set_device(local_rank)
# MAGIC
# MAGIC if not dist.is_initialized():
# MAGIC     init_kwargs = {"backend": "nccl", "init_method": "env://"}
# MAGIC     sig = inspect.signature(dist.init_process_group)
# MAGIC     if "device_id" in sig.parameters:
# MAGIC         init_kwargs["device_id"] = torch.device("cuda", local_rank)
# MAGIC     dist.init_process_group(**init_kwargs)
# MAGIC
# MAGIC global_rank = dist.get_rank()
# MAGIC world_size = dist.get_world_size()
# MAGIC is_global0 = (global_rank == 0)
# MAGIC ```
# MAGIC
# MAGIC #### 💡 Rank（ランク）とは？
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    分散学習における Rank                    │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
# MAGIC │  │   Node 0    │  │   Node 1    │  │   Node 2    │  │   Node 3    │
# MAGIC │  │             │  │             │  │             │  │             │
# MAGIC │  │ local_rank  │  │ local_rank  │  │ local_rank  │  │ local_rank  │
# MAGIC │  │     = 0     │  │     = 0     │  │     = 0     │  │     = 0     │
# MAGIC │  │             │  │             │  │             │  │             │
# MAGIC │  │ global_rank │  │ global_rank │  │ global_rank │  │ global_rank │
# MAGIC │  │     = 0     │  │     = 1     │  │     = 2     │  │     = 3     │
# MAGIC │  │   ★リーダー │  │             │  │             │  │             │
# MAGIC │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
# MAGIC │                                                             │
# MAGIC │  world_size = 4（全GPU数）                                  │
# MAGIC │                                                             │
# MAGIC │  ★ global_rank == 0 のGPUが「リーダー」                     │
# MAGIC │    → ログ出力、モデル保存などを担当                         │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC #### 💡 NCCL（ニックル）とは？
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │  NCCL = NVIDIA Collective Communications Library            │
# MAGIC │                                                             │
# MAGIC │  GPU同士が高速にデータをやり取りするためのライブラリ        │
# MAGIC │                                                             │
# MAGIC │    GPU 0 ←──────────────→ GPU 1                             │
# MAGIC │      ↑                      ↑                               │
# MAGIC │      │    NCCL が仲介       │                               │
# MAGIC │      ↓                      ↓                               │
# MAGIC │    GPU 2 ←──────────────→ GPU 3                             │
# MAGIC │                                                             │
# MAGIC │  勾配の集約（AllReduce）などを高速に行う                    │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 7. データセットの準備
# MAGIC
# MAGIC ```python
# MAGIC def build_user_text(ex):
# MAGIC     inst = (ex.get("instruction") or "").strip()
# MAGIC     inp = (ex.get("input") or "").strip()
# MAGIC     return f"{inst}\n\n[入力]\n{inp}" if inp else inst
# MAGIC
# MAGIC def to_text(ex):
# MAGIC     messages = [
# MAGIC         {"role": "system", "content": "/no_think"},
# MAGIC         {"role": "user", "content": build_user_text(ex)},
# MAGIC         {"role": "assistant", "content": (ex.get("output") or "").strip()},
# MAGIC     ]
# MAGIC     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
# MAGIC     return {"text": text}
# MAGIC ```
# MAGIC
# MAGIC #### 💡 データの変換イメージ
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    元のデータセット                         │
# MAGIC │  ┌───────────────────────────────────────────────────────┐ │
# MAGIC │  │ {                                                     │ │
# MAGIC │  │   "instruction": "日本の首都はどこですか？",          │ │
# MAGIC │  │   "input": "",                                        │ │
# MAGIC │  │   "output": "東京でござる"                            │ │
# MAGIC │  │ }                                                     │ │
# MAGIC │  └───────────────────────────────────────────────────────┘ │
# MAGIC │                          ↓                                  │
# MAGIC │                    to_text() で変換                         │
# MAGIC │                          ↓                                  │
# MAGIC │  ┌───────────────────────────────────────────────────────┐ │
# MAGIC │  │ チャット形式のテキスト                                │ │
# MAGIC │  │                                                       │ │
# MAGIC │  │ <|system|>/no_think<|end|>                            │ │
# MAGIC │  │ <|user|>日本の首都はどこですか？<|end|>               │ │
# MAGIC │  │ <|assistant|>東京でござる<|end|>                      │ │
# MAGIC │  └───────────────────────────────────────────────────────┘ │
# MAGIC │                                                             │
# MAGIC │  → モデルが理解できる形式に変換                             │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 8. DeepSpeed ZeRO-2 設定
# MAGIC
# MAGIC ```python
# MAGIC ds_config = {
# MAGIC     "wall_clock_breakdown": True,
# MAGIC     "train_micro_batch_size_per_gpu": "auto",
# MAGIC     "gradient_accumulation_steps": "auto",
# MAGIC     "bf16": {"enabled": True},
# MAGIC     "zero_optimization": {
# MAGIC         "stage": 2,
# MAGIC         "overlap_comm": True,
# MAGIC         "contiguous_gradients": True,
# MAGIC         "reduce_scatter": True,
# MAGIC     },
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC #### 💡 ZeRO（Zero Redundancy Optimizer）とは？
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    ZeRO の段階                              │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │ ZeRO Stage 0（通常のデータ並列）                    │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │   GPU 0        GPU 1        GPU 2        GPU 3      │   │
# MAGIC │  │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐      │   │
# MAGIC │  │  │Model │    │Model │    │Model │    │Model │      │   │
# MAGIC │  │  │Optim │    │Optim │    │Optim │    │Optim │      │   │
# MAGIC │  │  │Grad  │    │Grad  │    │Grad  │    │Grad  │      │   │
# MAGIC │  │  └──────┘    └──────┘    └──────┘    └──────┘      │   │
# MAGIC │  │  → 全GPU に同じものを複製（メモリ無駄遣い）         │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                          ↓                                  │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │ ZeRO Stage 1（Optimizer State分割）                 │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │   GPU 0        GPU 1        GPU 2        GPU 3      │   │
# MAGIC │  │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐      │   │
# MAGIC │  │  │Model │    │Model │    │Model │    │Model │      │   │
# MAGIC │  │  │Opt/4 │    │Opt/4 │    │Opt/4 │    │Opt/4 │      │   │
# MAGIC │  │  │Grad  │    │Grad  │    │Grad  │    │Grad  │      │   │
# MAGIC │  │  └──────┘    └──────┘    └──────┘    └──────┘      │   │
# MAGIC │  │  → Optimizer状態を4分割してメモリ節約               │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                          ↓                                  │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │ ZeRO Stage 2（+ Gradient分割）← 今回使用！          │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │   GPU 0        GPU 1        GPU 2        GPU 3      │   │
# MAGIC │  │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐      │   │
# MAGIC │  │  │Model │    │Model │    │Model │    │Model │      │   │
# MAGIC │  │  │Opt/4 │    │Opt/4 │    │Opt/4 │    │Opt/4 │      │   │
# MAGIC │  │  │Grd/4 │    │Grd/4 │    │Grd/4 │    │Grd/4 │      │   │
# MAGIC │  │  └──────┘    └──────┘    └──────┘    └──────┘      │   │
# MAGIC │  │  → Gradientも分割してさらにメモリ節約               │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                          ↓                                  │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │ ZeRO Stage 3（+ Model Parameter分割）               │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │   GPU 0        GPU 1        GPU 2        GPU 3      │   │
# MAGIC │  │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐      │   │
# MAGIC │  │  │Mdl/4 │    │Mdl/4 │    │Mdl/4 │    │Mdl/4 │      │   │
# MAGIC │  │  │Opt/4 │    │Opt/4 │    │Opt/4 │    │Opt/4 │      │   │
# MAGIC │  │  │Grd/4 │    │Grd/4 │    │Grd/4 │    │Grd/4 │      │   │
# MAGIC │  │  └──────┘    └──────┘    └──────┘    └──────┘      │   │
# MAGIC │  │  → 全て分割（最大のメモリ節約、通信コスト増）       │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC #### 💡 ZeRO-2 設定パラメータの解説
# MAGIC
# MAGIC | パラメータ | 説明 |
# MAGIC |-----------|------|
# MAGIC | `stage: 2` | ZeRO Stage 2を使用（Optimizer + Gradient分割） |
# MAGIC | `overlap_comm: True` | 通信と計算を同時に行い高速化 |
# MAGIC | `contiguous_gradients: True` | 勾配をメモリ上で連続配置して効率化 |
# MAGIC | `reduce_scatter: True` | 勾配集約を効率的に行う |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 9. SFTConfig（学習設定）
# MAGIC
# MAGIC ```python
# MAGIC args = SFTConfig(
# MAGIC     output_dir=output_dir,
# MAGIC     num_train_epochs=1,
# MAGIC     per_device_train_batch_size=1,
# MAGIC     gradient_accumulation_steps=8,
# MAGIC     learning_rate=2e-4,
# MAGIC     warmup_ratio=0.03,
# MAGIC     lr_scheduler_type="cosine",
# MAGIC     logging_steps=10,
# MAGIC     save_steps=200,
# MAGIC     save_total_limit=2,
# MAGIC     bf16=True,
# MAGIC     deepspeed=ds_config_path,
# MAGIC     optim="adamw_torch_fused",
# MAGIC     report_to=[],
# MAGIC     max_length=2048,
# MAGIC     packing=False,
# MAGIC     disable_tqdm=(not is_global0),
# MAGIC     gradient_checkpointing_kwargs={"use_reentrant": False},
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### 💡 主要パラメータの解説
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    学習パラメータ解説                       │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │                                                             │
# MAGIC │  【バッチサイズ関連】                                       │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │ per_device_train_batch_size = 1                     │   │
# MAGIC │  │   → 1つのGPUで一度に処理するサンプル数              │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │ gradient_accumulation_steps = 8                     │   │
# MAGIC │  │   → 8回分の勾配を貯めてからパラメータ更新           │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │ 実効バッチサイズ = 1 × 8 × 4(GPU) = 32              │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                                                             │
# MAGIC │  【学習率関連】                                             │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │ learning_rate = 2e-4 (0.0002)                       │   │
# MAGIC │  │   → パラメータ更新の大きさ                          │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │ warmup_ratio = 0.03                                 │   │
# MAGIC │  │   → 最初の3%のステップで学習率を徐々に上げる        │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │ lr_scheduler_type = "cosine"                        │   │
# MAGIC │  │   → コサインカーブで学習率を減衰                    │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                                                             │
# MAGIC │  【学習率スケジュールのイメージ】                           │
# MAGIC │                                                             │
# MAGIC │  学習率                                                     │
# MAGIC │    ↑                                                        │
# MAGIC │    │    ╭───────╮                                          │
# MAGIC │    │   ╱         ╲                                         │
# MAGIC │    │  ╱           ╲                                        │
# MAGIC │    │ ╱             ╲                                       │
# MAGIC │    │╱               ╲                                      │
# MAGIC │    └──────────────────→ ステップ                           │
# MAGIC │     ↑warmup    cosine decay                                │
# MAGIC │                                                             │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC #### 💡 その他の重要パラメータ
# MAGIC
# MAGIC | パラメータ | 値 | 説明 |
# MAGIC |-----------|-----|------|
# MAGIC | `num_train_epochs` | 1 | データセット全体を1回学習 |
# MAGIC | `bf16` | True | BFloat16精度で学習（メモリ節約＆高速化） |
# MAGIC | `max_length` | 2048 | 入力テキストの最大トークン数 |
# MAGIC | `save_steps` | 200 | 200ステップごとにチェックポイント保存 |
# MAGIC | `save_total_limit` | 2 | 保存するチェックポイントは最大2つ |
# MAGIC | `logging_steps` | 10 | 10ステップごとにログ出力 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 10. MLflowLoggingCallback（カスタムコールバック）
# MAGIC
# MAGIC ```python
# MAGIC class MLflowLoggingCallback(TrainerCallback):
# MAGIC     def on_train_begin(self, args, state, control, **kwargs):
# MAGIC         """トレーニング開始時"""
# MAGIC     
# MAGIC     def on_log(self, args, state, control, logs=None, **kwargs):
# MAGIC         """ログ出力時（logging_stepsごと）"""
# MAGIC     
# MAGIC     def on_save(self, args, state, control, **kwargs):
# MAGIC         """チェックポイント保存時"""
# MAGIC     
# MAGIC     def on_train_end(self, args, state, control, **kwargs):
# MAGIC         """トレーニング終了時"""
# MAGIC     
# MAGIC     def on_evaluate(self, args, state, control, metrics=None, **kwargs):
# MAGIC         """評価時"""
# MAGIC ```
# MAGIC
# MAGIC #### 💡 コールバックの動作タイミング
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    学習ループとコールバック                 │
# MAGIC │                                                             │
# MAGIC │  trainer.train() 開始                                       │
# MAGIC │         │                                                   │
# MAGIC │         ▼                                                   │
# MAGIC │  ┌─────────────────────┐                                   │
# MAGIC │  │  on_train_begin()   │ ← 学習開始時に1回                 │
# MAGIC │  └─────────────────────┘                                   │
# MAGIC │         │                                                   │
# MAGIC │         ▼                                                   │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │              学習ループ                             │   │
# MAGIC │  │  ┌─────────────────────────────────────────────┐   │   │
# MAGIC │  │  │  Step 1, 2, 3, ... 10                       │   │   │
# MAGIC │  │  │         │                                   │   │   │
# MAGIC │  │  │         ▼                                   │   │   │
# MAGIC │  │  │  ┌─────────────┐                           │   │   │
# MAGIC │  │  │  │  on_log()   │ ← 10ステップごと          │   │   │
# MAGIC │  │  │  └─────────────┘                           │   │   │
# MAGIC │  │  └─────────────────────────────────────────────┘   │   │
# MAGIC │  │                    ...                              │   │
# MAGIC │  │  ┌─────────────────────────────────────────────┐   │   │
# MAGIC │  │  │  Step 191, 192, ... 200                     │   │   │
# MAGIC │  │  │         │                                   │   │   │
# MAGIC │  │  │         ▼                                   │   │   │
# MAGIC │  │  │  ┌─────────────┐  ┌─────────────┐          │   │   │
# MAGIC │  │  │  │  on_log()   │  │  on_save()  │          │   │   │
# MAGIC │  │  │  └─────────────┘  └─────────────┘          │   │   │
# MAGIC │  │  │                    ↑ 200ステップごと        │   │   │
# MAGIC │  │  └─────────────────────────────────────────────┘   │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │         │                                                   │
# MAGIC │         ▼                                                   │
# MAGIC │  ┌─────────────────────┐                                   │
# MAGIC │  │   on_train_end()    │ ← 学習終了時に1回                 │
# MAGIC │  └─────────────────────┘                                   │
# MAGIC │                                                             │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC #### 💡 MLflowに記録される情報
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    MLflow Tracking                          │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │  Experiment: nemotron_nano_gozaru_fullft_mn         │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │  Run: nemotron_nano_9b_gozaru_fullft_sft_zero2      │   │
# MAGIC │  │  ┌─────────────────────────────────────────────┐   │   │
# MAGIC │  │  │  Tags:                                      │   │   │
# MAGIC │  │  │    • base_model: nvidia/Nemotron-Nano-9B    │   │   │
# MAGIC │  │  │    • dataset: databricks-dolly-15k-ja-gozaru│   │   │
# MAGIC │  │  │    • task: SFT full-parameter finetuning    │   │   │
# MAGIC │  │  ├─────────────────────────────────────────────┤   │   │
# MAGIC │  │  │  Parameters:                                │   │   │
# MAGIC │  │  │    • epochs: 1                              │   │   │
# MAGIC │  │  │    • per_device_train_batch_size: 1         │   │   │
# MAGIC │  │  │    • grad_accum: 8                          │   │   │
# MAGIC │  │  │    • lr: 0.0002                             │   │   │
# MAGIC │  │  ├─────────────────────────────────────────────┤   │   │
# MAGIC │  │  │  Metrics (時系列):                          │   │   │
# MAGIC │  │  │    • loss: 2.5 → 1.8 → 1.2 → ...           │   │   │
# MAGIC │  │  │    • learning_rate: 0 → 0.0002 → ...       │   │   │
# MAGIC │  │  │    • time_per_step: 3.2s, 3.1s, ...        │   │   │
# MAGIC │  │  ├─────────────────────────────────────────────┤   │   │
# MAGIC │  │  │  Artifacts:                                 │   │   │
# MAGIC │  │  │    📁 model/                                │   │   │
# MAGIC │  │  │       ├── config.json                       │   │   │
# MAGIC │  │  │       ├── model.safetensors                 │   │   │
# MAGIC │  │  │       └── tokenizer.json                    │   │   │
# MAGIC │  │  │    📁 checkpoints/                          │   │   │
# MAGIC │  │  │       ├── checkpoint-200/                   │   │   │
# MAGIC │  │  │       └── checkpoint-400/                   │   │   │
# MAGIC │  │  │    📁 logs/                                 │   │   │
# MAGIC │  │  │       └── training_output.log               │   │   │
# MAGIC │  │  └─────────────────────────────────────────────┘   │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 11. メインセル（実行部分）
# MAGIC
# MAGIC ```python
# MAGIC # MLflow実験の設定
# MAGIC username = spark.sql("SELECT current_user()").collect()[0][0]
# MAGIC MLFLOW_EXPERIMENT_NAME = f"/Workspace/Users/{username}/nemotron_nano_gozaru_fullft_mn"
# MAGIC
# MAGIC if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
# MAGIC     mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME)
# MAGIC mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
# MAGIC
# MAGIC # MLflow Runの開始と分散学習の実行
# MAGIC with mlflow.start_run(run_name="nemotron_nano_9b_gozaru_fullft_sft_zero2") as run:
# MAGIC     # タグとパラメータの記録
# MAGIC     mlflow.set_tag("base_model", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
# MAGIC     mlflow.set_tag("dataset", "bbz662bbz/databricks-dolly-15k-ja-gozaru")
# MAGIC     
# MAGIC     mlflow.log_params({
# MAGIC         "epochs": 1,
# MAGIC         "per_device_train_batch_size": 1,
# MAGIC         "grad_accum": 8,
# MAGIC         "lr": 2e-4,
# MAGIC         "nnodes": NNODES,
# MAGIC         "gpus_per_node": NUM_GPUS_PER_NODE,
# MAGIC     })
# MAGIC
# MAGIC     # DeepSpeed Distributorの設定と実行
# MAGIC     distributor = DeepspeedTorchDistributor(
# MAGIC         numGpus=NUM_GPUS_PER_NODE,
# MAGIC         nnodes=NNODES,
# MAGIC         localMode=False,
# MAGIC         useGpu=True,
# MAGIC         deepspeedConfig=None,
# MAGIC     )
# MAGIC
# MAGIC     result = distributor.run(
# MAGIC         train_nemotron_fullft_zero2, 
# MAGIC         host=DATABRICKS_HOST, 
# MAGIC         token=DATABRICKS_TOKEN, 
# MAGIC         exp_name=MLFLOW_EXPERIMENT_NAME, 
# MAGIC         run_id=run.info.run_id
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC #### 💡 実行フローの全体像
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    実行フロー全体像                         │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │  Driver Node（ノートブック実行環境）                │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │  1. MLflow Experiment作成/設定                      │   │
# MAGIC │  │  2. MLflow Run開始                                  │   │
# MAGIC │  │  3. タグ・パラメータ記録                            │   │
# MAGIC │  │  4. DeepspeedTorchDistributor作成                   │   │
# MAGIC │  │  5. distributor.run() 呼び出し                      │   │
# MAGIC │  │         │                                           │   │
# MAGIC │  └─────────│───────────────────────────────────────────┘   │
# MAGIC │            │                                                │
# MAGIC │            │ 学習関数と引数を各Workerに配布                 │
# MAGIC │            ▼                                                │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │              Worker Nodes（学習実行環境）           │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
# MAGIC │  │  │ Worker 0  │ │ Worker 1  │ │ Worker 2  │ │ Worker 3  │
# MAGIC │  │  │ (rank 0)  │ │ (rank 1)  │ │ (rank 2)  │ │ (rank 3)  │
# MAGIC │  │  │  ★リーダー │ │           │ │           │ │           │
# MAGIC │  │  │           │ │           │ │           │ │           │
# MAGIC │  │  │ ・ログ出力│ │ ・学習のみ│ │ ・学習のみ│ │ ・学習のみ│
# MAGIC │  │  │ ・モデル  │ │           │ │           │ │           │
# MAGIC │  │  │   保存    │ │           │ │           │ │           │
# MAGIC │  │  │ ・MLflow  │ │           │ │           │ │           │
# MAGIC │  │  │   記録    │ │           │ │           │ │           │
# MAGIC │  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘
# MAGIC │  │        ↑               ↑           ↑           ↑        │
# MAGIC │  │        └───────────────┴───────────┴───────────┘        │
# MAGIC │  │                    NCCL通信                             │
# MAGIC │  │              （勾配の同期・集約）                       │
# MAGIC │  │                                                     │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │            │                                                │
# MAGIC │            │ 学習完了後、結果を返す                         │
# MAGIC │            ▼                                                │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │  Driver Node                                        │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │  6. result を受け取る                               │   │
# MAGIC │  │  7. "✅ All done!" を出力                           │   │
# MAGIC │  │                                                     │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 重要な概念のまとめ
# MAGIC
# MAGIC ### SFT（Supervised Fine-Tuning）とは
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    SFT の概念                               │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │  事前学習済みモデル（Nemotron-Nano-9B）             │   │
# MAGIC │  │  ┌─────────────────────────────────────────────┐   │   │
# MAGIC │  │  │  大量のテキストで学習済み                   │   │   │
# MAGIC │  │  │  → 一般的な言語能力を持っている             │   │   │
# MAGIC │  │  │  → でも特定のタスクに最適化されていない     │   │   │
# MAGIC │  │  └─────────────────────────────────────────────┘   │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                          │                                  │
# MAGIC │                          │ SFT（教師あり微調整）            │
# MAGIC │                          │ 「こう聞かれたら、こう答える」   │
# MAGIC │                          │ という例を見せて学習             │
# MAGIC │                          ▼                                  │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │  ファインチューニング済みモデル                     │   │
# MAGIC │  │  ┌─────────────────────────────────────────────┐   │   │
# MAGIC │  │  │  特定のタスク・スタイルに最適化              │   │   │
# MAGIC │  │  │  → 「〜でござる」口調で回答できる！         │   │   │
# MAGIC │  │  └─────────────────────────────────────────────┘   │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Full Fine-Tuning vs LoRA/QLoRA
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │              ファインチューニング手法の比較                 │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │  Full Fine-Tuning（今回の手法）                     │   │
# MAGIC │  │  ┌─────────────────────────────────────────────┐   │   │
# MAGIC │  │  │  モデルの全パラメータを更新                 │   │   │
# MAGIC │  │  │                                             │   │   │
# MAGIC │  │  │  ✅ メリット                                │   │   │
# MAGIC │  │  │    • 最高の性能を引き出せる可能性           │   │   │
# MAGIC │  │  │    • 大きな変更が可能                       │   │   │
# MAGIC │  │  │                                             │   │   │
# MAGIC │  │  │  ❌ デメリット                              │   │   │
# MAGIC │  │  │    • 大量のGPUメモリが必要                  │   │   │
# MAGIC │  │  │    • 学習時間が長い                         │   │   │
# MAGIC │  │  │    • DeepSpeed等の分散学習が必須            │   │   │
# MAGIC │  │  └─────────────────────────────────────────────┘   │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │  LoRA / QLoRA（参考）                               │   │
# MAGIC │  │  ┌─────────────────────────────────────────────┐   │   │
# MAGIC │  │  │  一部のパラメータのみ更新                   │   │   │
# MAGIC │  │  │                                             │   │   │
# MAGIC │  │  │  ✅ メリット                                │   │   │
# MAGIC │  │  │    • 少ないGPUメモリで学習可能              │   │   │
# MAGIC │  │  │    • 学習が高速                             │   │   │
# MAGIC │  │  │                                             │   │   │
# MAGIC │  │  │  ❌ デメリット                              │   │   │
# MAGIC │  │  │    • Full FTより性能が劣る場合がある        │   │   │
# MAGIC │  │  └─────────────────────────────────────────────┘   │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 学習の流れ（時系列）
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    学習の時系列フロー                       │
# MAGIC │                                                             │
# MAGIC │  時間 →                                                     │
# MAGIC │  ════════════════════════════════════════════════════════   │
# MAGIC │                                                             │
# MAGIC │  [開始]                                                     │
# MAGIC │    │                                                        │
# MAGIC │    ▼                                                        │
# MAGIC │  ┌──────────────────────────────────────────────────────┐  │
# MAGIC │  │ 1. 環境初期化（〜1分）                               │  │
# MAGIC │  │    • NCCL初期化                                      │  │
# MAGIC │  │    • プロセスグループ作成                            │  │
# MAGIC │  │    • GPU間通信の確立                                 │  │
# MAGIC │  └──────────────────────────────────────────────────────┘  │
# MAGIC │    │                                                        │
# MAGIC │    ▼                                                        │
# MAGIC │  ┌──────────────────────────────────────────────────────┐  │
# MAGIC │  │ 2. モデル・データ読み込み（〜5分）                   │  │
# MAGIC │  │    • Hugging Faceからモデルダウンロード              │  │
# MAGIC │  │    • トークナイザー読み込み                          │  │
# MAGIC │  │    • データセット読み込み・前処理                    │  │
# MAGIC │  └──────────────────────────────────────────────────────┘  │
# MAGIC │    │                                                        │
# MAGIC │    ▼                                                        │
# MAGIC │  ┌──────────────────────────────────────────────────────┐  │
# MAGIC │  │ 3. 学習ループ（数時間〜）                            │  │
# MAGIC │  │                                                      │  │
# MAGIC │  │    Step 10:  loss=2.5, lr=0.00005  ← on_log()       │  │
# MAGIC │  │    Step 20:  loss=2.3, lr=0.00010                    │  │
# MAGIC │  │    ...                                               │  │
# MAGIC │  │    Step 200: loss=1.8, lr=0.00020  ← on_save()      │  │
# MAGIC │  │              📁 checkpoint-200 保存                  │  │
# MAGIC │  │    ...                                               │  │
# MAGIC │  │    Step 400: loss=1.5, lr=0.00018  ← on_save()      │  │
# MAGIC │  │              📁 checkpoint-400 保存                  │  │
# MAGIC │  │    ...                                               │  │
# MAGIC │  │    Step N:   loss=1.2, lr=0.00001  ← 学習終了       │  │
# MAGIC │  │                                                      │  │
# MAGIC │  └──────────────────────────────────────────────────────┘  │
# MAGIC │    │                                                        │
# MAGIC │    ▼                                                        │
# MAGIC │  ┌──────────────────────────────────────────────────────┐  │
# MAGIC │  │ 4. 保存・アップロード（〜10分）                      │  │
# MAGIC │  │    • 最終モデルを保存                                │  │
# MAGIC │  │    • MLflowにアップロード                            │  │
# MAGIC │  │    • ログファイルをアップロード                      │  │
# MAGIC │  └──────────────────────────────────────────────────────┘  │
# MAGIC │    │                                                        │
# MAGIC │    ▼                                                        │
# MAGIC │  [完了] ✅ All done!                                        │
# MAGIC │                                                             │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠️ トラブルシューティング
# MAGIC
# MAGIC ### よくある問題と対処法
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    トラブルシューティング                   │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │ 問題1: CUDA Out of Memory                           │   │
# MAGIC │  │ ─────────────────────────────────────────────────── │   │
# MAGIC │  │ 症状: RuntimeError: CUDA out of memory              │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │ 対処法:                                             │   │
# MAGIC │  │   • per_device_train_batch_size を下げる（1→1で既に最小）│
# MAGIC │  │   • gradient_accumulation_steps を上げる            │   │
# MAGIC │  │   • max_length を短くする（2048→1024）              │   │
# MAGIC │  │   • ZeRO Stage 3 に変更する                         │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │ 問題2: NCCL Timeout                                 │   │
# MAGIC │  │ ─────────────────────────────────────────────────── │   │
# MAGIC │  │ 症状: NCCL timeout / connection refused             │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │ 対処法:                                             │   │
# MAGIC │  │   • NCCL_DEBUG=INFO で詳細ログを確認                │   │
# MAGIC │  │   • NCCL_SOCKET_IFNAME を正しいNICに設定            │   │
# MAGIC │  │   • ファイアウォール設定を確認                      │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                                                             │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │ 問題3: MLflow接続エラー                             │   │
# MAGIC │  │ ─────────────────────────────────────────────────── │   │
# MAGIC │  │ 症状: MLflow logging failed                         │   │
# MAGIC │  │                                                     │   │
# MAGIC │  │ 対処法:                                             │   │
# MAGIC │  │   • DATABRICKS_HOST, DATABRICKS_TOKEN を確認        │   │
# MAGIC │  │   • ネットワーク接続を確認                          │   │
# MAGIC │  │   • MLflow Tracking URIを確認                       │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                                                             │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 用語集
# MAGIC
# MAGIC | 用語 | 説明 |
# MAGIC |------|------|
# MAGIC | **LLM** | Large Language Model（大規模言語モデル） |
# MAGIC | **SFT** | Supervised Fine-Tuning（教師あり微調整） |
# MAGIC | **DeepSpeed** | Microsoftが開発した分散学習ライブラリ |
# MAGIC | **ZeRO** | Zero Redundancy Optimizer（メモリ効率化技術） |
# MAGIC | **NCCL** | NVIDIA Collective Communications Library（GPU間通信） |
# MAGIC | **MLflow** | 機械学習の実験管理ツール |
# MAGIC | **Rank** | 分散学習における各プロセスの識別番号 |
# MAGIC | **Gradient Accumulation** | 勾配を複数ステップ分貯めてから更新する手法 |
# MAGIC | **BFloat16 (bf16)** | 16ビット浮動小数点形式（メモリ節約） |
# MAGIC | **Checkpoint** | 学習途中のモデル状態の保存 |
# MAGIC | **Tokenizer** | テキストをモデルが理解できる数値に変換するツール |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 まとめ
# MAGIC
# MAGIC このコードは以下のことを実現しています：
# MAGIC
# MAGIC 1. **大規模モデルの分散学習**: 9Bパラメータのモデルを4つのGPUで効率的に学習
# MAGIC 2. **メモリ効率化**: DeepSpeed ZeRO-2で限られたGPUメモリを有効活用
# MAGIC 3. **実験管理**: MLflowで学習の進捗・結果を記録
# MAGIC 4. **日本語対応**: 日本語データセットでファインチューニング
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    最終成果物                               │
# MAGIC │                                                             │
# MAGIC │  入力: 「日本で一番高い山は？」                             │
# MAGIC │                    ↓                                        │
# MAGIC │  ┌─────────────────────────────────────────────────────┐   │
# MAGIC │  │  ファインチューニング済み                           │   │
# MAGIC │  │  Nemotron-Nano-9B-Gozaru                            │   │
# MAGIC │  └─────────────────────────────────────────────────────┘   │
# MAGIC │                    ↓                                        │
# MAGIC │  出力: 「富士山でござる！標高3,776メートルの日本最高峰     │
# MAGIC │         でござるよ。」                                      │
# MAGIC │                                                             │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------


