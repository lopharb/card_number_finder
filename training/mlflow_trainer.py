from ultralytics.models.yolo.segment import SegmentationTrainer
import mlflow


class MLflowSegmentationTrainer(SegmentationTrainer):
    """
    A class extending the SegmentationTrainer class to add MLFlow tracking.
    """

    def before_train(self) -> None:
        """
        Log MLFlow params here (after everything is initialized properly)
        """
        mlflow.set_experiment("yolo-segmentation")
        mlflow.log_params({
            "epochs": self.args.epochs,
            "imgsz": self.args.imgsz,
            "batch": self.args.batch,
            "model": str(self.args.model),
        })
