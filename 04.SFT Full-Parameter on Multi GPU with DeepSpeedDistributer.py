# Databricks notebook source
# MAGIC %md
# MAGIC # 17.3 ML LTS

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

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
NNODES = int(os.environ.get("NNODES", "1"))
NUM_GPUS_PER_NODE = int(os.environ.get("NUM_GPUS_PER_NODE", "4"))

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
MLFLOW_EXPERIMENT_NAME = f"/Workspace/Users/{username}/nemotron_v2_nano_gozaru_fullft_single_node"

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
        localMode=True,
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


