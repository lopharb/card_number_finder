import os
from pathlib import Path
import yaml

from ultralytics.models.yolo.segment import SegmentationTrainer
import mlflow
from ultralytics.utils import yaml_load

from .card_dataset import CardSegmentationDataset

from ultralytics.models.yolo.segment import SegmentationTrainer
import mlflow


class MLflowSegmentationTrainer(SegmentationTrainer):
    def __init__(self, overrides=None, _callbacks=None):
        # Call the original constructor (DO NOT block cfg loading)
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        if isinstance(self.args.data, str):
            data_path = os.path.abspath(self.args.data)
            self.data = yaml_load(data_path)  # now self.data is guaranteed to exist
            print('loading yaml...')
        else:
            self.data = self.args.data  # already a dict
        print(self.data)

    def _get_paths(self, data_cfg_path):
        data_cfg_path_ = Path(data_cfg_path)
        with open(data_cfg_path_, "r") as f:
            data_cfg = yaml.safe_load(f)

        train_path = data_cfg_path_.parent / data_cfg["train"]
        val_path = data_cfg_path_.parent / data_cfg["val"]
        return train_path.parent, val_path.parent

    def get_dataset(self):
        """
        Override this only if you want custom dataset loading (like our on-the-fly augmentations).
        Otherwise, just use default and let YOLO handle it.
        """
        train_path, val_path = self._get_paths(self.args.data)
        trainset = CardSegmentationDataset(
            image_dir=os.path.join(train_path, 'images'),
            label_dir=os.path.join(train_path, 'labels')
        )
        valset = CardSegmentationDataset(
            image_dir=os.path.join(val_path, 'images'),
            label_dir=os.path.join(val_path, 'labels')
        )
        return trainset, valset

    def before_train(self):
        # Log MLFlow params here (after everything is initialized properly)
        mlflow.set_experiment("yolo-segmentation")
        mlflow.log_params({
            "epochs": self.args.epochs,
            "imgsz": self.args.imgsz,
            "batch": self.args.batch,
            "model": str(self.args.model),
        })
