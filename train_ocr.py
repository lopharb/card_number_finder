import mlflow
import subprocess
import time

mlflow.set_experiment("paddleocr-creditcard")

with mlflow.start_run():
    mlflow.log_params({
        "model": "CRNN",
        "backbone": "MobileNetV3",
        "algorithm": "CTC",
        "epochs": 50,
        "batch_size": 32
    })

    start = time.time()
    result = subprocess.run([
        "paddleocr",
        "tools/train.py",
        "-c", "data/synthetic_numbers/rec_train.yml"
    ], capture_output=True, text=True)

    mlflow.log_metric("train_time_sec", time.time() - start)

    # Log stdout/stderr for debugging
    with open("train_log.txt", "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)
    mlflow.log_artifact("train_log.txt")

    # Log final model
    # mlflow.log_artifacts("./output/ocr_model", artifact_path="ocr_model")
