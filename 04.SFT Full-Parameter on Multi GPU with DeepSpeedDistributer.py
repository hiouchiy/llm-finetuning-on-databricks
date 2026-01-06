# Databricks notebook source
# MAGIC %md
# MAGIC # 17.3 ML LTS

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor
import torch

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

GPU_NUM = torch.cuda.device_count()

# ========================================
# Databricks認証情報を取得して環境変数に設定
# ========================================
def get_databricks_credentials():
    """ノートブック環境からDatabricks認証情報を取得"""
    try:
        # Databricksノートブック環境では dbutils が利用可能
        from dbruntime.databricks_repl_context import get_context
        context = get_context()
        
        host = context.apiUrl  # 例: "https://xxx.cloud.databricks.com"
        token = context.apiToken
        
        return host, token
    except Exception as e:
        print(f"Warning: Could not get credentials automatically: {e}")
        return None, None

# 認証情報を取得して環境変数に設定
DATABRICKS_HOST, DATABRICKS_TOKEN = get_databricks_credentials()

if DATABRICKS_HOST and DATABRICKS_TOKEN:
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
    print(f"✅ Set DATABRICKS_HOST: {DATABRICKS_HOST}")
else:
    print("⚠️ Could not retrieve Databricks credentials")

# ========================================
# 標準出力を画面とファイルに同時出力するクラス
# ========================================
class TeeLogger:
    """標準出力を画面とファイルに同時に書き込む"""
    def __init__(self, log_file, mode='a'):
        self.terminal = sys.stdout
        self.log = open(log_file, mode, buffering=1)  # 行バッファリング
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # ★ 即座にディスクに書き込み
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


# ========================================
# トレーニング関数（MLflowコールバック復活）
# ========================================
def train_nemotron_fullft():
    import os
    import sys
    import torch
    import torch.distributed as dist
    import mlflow
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from trl import SFTTrainer
    
    # 1) local_rank を取る（torchrun/Distributor が環境変数で渡す）
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # 2) まず “このプロセスが使うGPU” を固定（ここがポイント）
    torch.cuda.set_device(local_rank)
    
    # DDP初期化
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)
    os.environ["NCCL_P2P_DISABLE"] = "1"

    import os
    import torch
    import torch.distributed as dist
    import mlflow
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from trl import SFTTrainer, SFTConfig
    
    # DDP初期化
    # dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    # 環境変数からMLflow Run IDを取得
    mlflow_run_id = os.environ.get("MLFLOW_RUN_ID")

    # ★ rank 0のみログファイルに記録
    log_file_path = "/local_disk0/training_output.log"
    if local_rank == 0:
        # 標準出力を画面とファイルに分岐
        tee = TeeLogger(log_file_path, mode='w')  # 'w'で新規作成
        sys.stdout = tee
        sys.stderr = tee  # エラーも記録したい場合
        print(f"📝 Logging all output to {log_file_path}")
    
    # ========================================
    # MLflowコールバック（rank 0のみ）
    # ========================================
    class MLflowLoggingCallback(TrainerCallback):
        def __init__(self, run_id, log_file):
            self.run_id = run_id
            self.log_file = log_file
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.last_log_time = None
            self.upload_threads = []
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.local_rank == 0 and logs and self.run_id:
                try:
                    import time
                    current_time = time.time()
                    
                    if self.last_log_time is not None:
                        elapsed = current_time - self.last_log_time
                        time_per_step = elapsed / args.logging_steps
                        logs["time_per_step"] = time_per_step
                    
                    self.last_log_time = current_time
                    
                    with mlflow.start_run(run_id=self.run_id):
                        for key, value in logs.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(key, value, step=state.global_step)
                        
                        # ★ ログファイルも定期的にアップロード
                        if state.global_step % 100 == 0:  # 100ステップごと
                            mlflow.log_artifact(self.log_file, artifact_path="logs")
                            print(f"📤 Uploaded training log (step {state.global_step})")
                
                except Exception as e:
                    print(f"Warning: MLflow logging failed at step {state.global_step}: {e}")
        
        def on_save(self, args, state, control, **kwargs):
            """チェックポイント保存時に非同期アップロード"""
            if self.local_rank == 0 and self.run_id:
                import threading
                import os
                
                checkpoint_folder = f"checkpoint-{state.global_step}"
                checkpoint_path = os.path.join(args.output_dir, checkpoint_folder)
                
                if not os.path.exists(checkpoint_path):
                    return
                
                def upload_checkpoint():
                    try:
                        import mlflow
                        with mlflow.start_run(run_id=self.run_id):
                            print(f"📤 Uploading {checkpoint_folder} to MLflow (async)...")
                            mlflow.log_artifacts(
                                checkpoint_path,
                                artifact_path=f"checkpoints/{checkpoint_folder}"
                            )
                            print(f"✅ {checkpoint_folder} uploaded to MLflow")
                    except Exception as e:
                        print(f"❌ Checkpoint upload failed for {checkpoint_folder}: {e}")
                
                thread = threading.Thread(target=upload_checkpoint, daemon=False)
                thread.start()
                self.upload_threads.append(thread)
                self.upload_threads = [t for t in self.upload_threads if t.is_alive()]
        
        def on_train_end(self, args, state, control, **kwargs):
            """学習終了時に全アップロードを待機＋最終ログをアップロード"""
            if self.local_rank == 0:
                if self.upload_threads:
                    print(f"⏳ Waiting for {len(self.upload_threads)} uploads to complete...")
                    for thread in self.upload_threads:
                        thread.join()
                    print("✅ All checkpoint uploads completed")
                
                # ★ 最終ログをアップロード
                if self.run_id:
                    try:
                        import mlflow
                        with mlflow.start_run(run_id=self.run_id):
                            mlflow.log_artifact(self.log_file, artifact_path="logs")
                            print(f"✅ Final training log uploaded to MLflow")
                    except Exception as e:
                        print(f"Warning: Final log upload failed: {e}")
        
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if self.local_rank == 0 and metrics and self.run_id:
                try:
                    with mlflow.start_run(run_id=self.run_id):
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"eval_{key}", value, step=state.global_step)
                except Exception as e:
                    print(f"Warning: MLflow eval logging failed: {e}")

    
    MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    DATASET_ID = "bbz662bbz/databricks-dolly-15k-ja-gozaru"
    
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
    
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.map(to_text, remove_columns=ds.column_names)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(local_rank)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    output_dir = "/local_disk0/nemotron_nano_9b_gozaru_fullft"
    

    # ===============================
    # DeepSpeed ZeRO-3 config (CPU offloadなし)
    # - ZeRO Stage 3
    # - offload_param / offload_optimizer は設定しない（=CPU offloadなし）
    # - stage3_gather_16bit_weights_on_model_save を有効化（保存時の全GPU集約）
    # ===============================
    import json
    # ds_config = {
    #     "train_micro_batch_size_per_gpu": "auto",
    #     "gradient_accumulation_steps": "auto",
    #     "bf16": {"enabled": True},
    #     "zero_optimization": {
    #         "stage": 3,
    #         "overlap_comm": True,
    #         "contiguous_gradients": True,
    #         "reduce_scatter": True,
    #         "stage3_gather_16bit_weights_on_model_save": True,
    #         # "zero_allow_untested_optimizer": True
    #     }
    # }
    
    # ds_config_path = "/tmp/ds_zero3_no_offload.json"
    
    ds_config = {
        "wall_clock_breakdown": True,
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,  # Stage 3 → 2 に変更
            "offload_optimizer": {
                "device": "none"  # ✅ 辞書形式で指定
            },
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_bucket_size": 2e8,
            "allgather_bucket_size": 2e8
        }
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
        logging_steps=1,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        deepspeed=ds_config_path,
        optim="adamw_torch_fused",
        report_to=[],
        max_length=2048,
        packing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # ddp_find_unused_parameters=False,
    )
    
    # コールバックを追加
    callbacks = []
    if mlflow_run_id:
        callbacks.append(MLflowLoggingCallback(run_id=mlflow_run_id, log_file=log_file_path))
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=args,
        callbacks=callbacks,
    )
    
    # 学習実行
    train_result = trainer.train()
    
    # rank 0のみモデル保存
    if local_rank == 0:
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("✅ Training done")
        print("model_dir:", output_dir)

        # ★ 標準出力を元に戻す
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.close()
    
    # dist.destroy_process_group()
    
    if local_rank == 0:
        return {"model_dir": output_dir, "log_file": log_file_path, "metrics": train_result.metrics}
    return None


# ========================================
# メインセル：MLflow Run作成 → 学習実行
# ========================================
import mlflow

username = spark.sql("SELECT current_user()").collect()[0][0]
MLFLOW_EXPERIMENT_NAME = f"/Workspace/Users/{username}/nemotron_nano_gozaru_fullft"

if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
    mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# MLflow configuration
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "TRUE"  # enable MLflow logging of artifacts
os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT_NAME

with mlflow.start_run(run_name="nemotron_nano_9b_gozaru_fullft_sft") as run:
    # パラメータ記録
    mlflow.set_tag("base_model", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    mlflow.set_tag("dataset", "bbz662bbz/databricks-dolly-15k-ja-gozaru")
    mlflow.set_tag("task", "SFT full-parameter finetuning (DeepSpeed ZeRO-3)")
    mlflow.log_params({        "epochs": 1,
        "per_device_train_batch_size": 1,
        "grad_accum": 8,
        "lr": 2e-4,
        "num_gpus": GPU_NUM,
    })
    
    # Run IDを環境変数に設定（子プロセスに引き継がれる）
    os.environ["MLFLOW_RUN_ID"] = run.info.run_id
    
    # DDP学習実行
    distributor = DeepspeedTorchDistributor(
        numGpus=GPU_NUM,
        nnodes=1,
        localMode=True,
        useGpu=True,
        deepspeedConfig=None  # ZeRO設定はSFTConfig(deepspeed=...)側で指定
    )
    
    result = distributor.run(train_nemotron_fullft)
    
    # アーティファクト保存
    if result and "model_dir" in result:
        mlflow.log_artifacts(result["model_dir"], artifact_path="model")

print("✅ All done!")

# COMMAND ----------


