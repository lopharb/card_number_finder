from ultralytics.engine.trainer import BaseTrainer
from torch.utils.data import DataLoader
import mlflow
from .collate import yolo_segmentation_collate


class MLflowSegmentationTrainer(BaseTrainer):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        dataset = train_dataset if mode == 'train' else val_dataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), collate_fn=yolo_segmentation_collate)

    def _do_mlflow_logging(self):
        mlflow.log_param("epochs", self.args.epochs)
        mlflow.log_param("imgsz", self.args.imgsz)
        mlflow.log_param("batch", self.args.batch)

    def _log_results(self):
        metrics = self.metrics
        mlflow.log_metric("metrics/mAP50-seg", metrics.box.map50)
        mlflow.log_metric("metrics/mAP-seg", metrics.box.map)
        mlflow.log_artifacts(str(self.save_dir / "weights"), artifact_path="model_weights")

    def train(self):
        with mlflow.start_run(run_name="YOLOv8-Segmentation"):
            self._do_mlflow_logging()
            super().train()
            self._log_results()
