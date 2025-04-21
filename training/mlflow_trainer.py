import mlflow
from ultralytics.models.yolo.segment import SegmentationTrainer


class MLflowSegmentationTrainer(SegmentationTrainer):
    """
    A class extending the SegmentationTrainer class to add MLFlow tracking.
    """

    def __init__(self, **kwargs):
        """
        Initialize the trainer and strat logging.
        """
        super().__init__(**kwargs)

        self.start_logging()

    def start_logging(self) -> None:
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

        # FIXME there should be a better way to do this
        augmentation_keys = [
            "hsv_h", "hsv_s", "hsv_v",
            "degrees", "translate", "scale", "shear", "perspective",
            "flipud", "fliplr", "mosaic", "mixup", "copy_paste"
        ]

        augmentations = {key: getattr(self.args, key, None) for key in augmentation_keys if hasattr(self.args, key)}
        mlflow.log_dict(augmentations, "augmentations.json")
