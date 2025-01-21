import os
import torch

from recipes.checkpoints.config import CheckpointConfig


class Checkpoint:
    def __init__(self, config: CheckpointConfig, output_dir: str = None):
        self.config = config
        self.output_dir = output_dir
        self._pipline()

    def _preprocess(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _prepare(self):
        self.metrics = {
            "rng": self.config.rng,
            "batch_size": self.config.batch_size, "step": self.config.step,
            "loss": self.config.loss, "best_loss": self.config.best_loss,
        }

    def _pipline(self):
        self._preprocess()
        self._prepare()

    def save(self) -> None:
        self.save_model(self.config.model)
        self.save_optimizer(self.config.optimizer)
        self.save_scheduler(self.config.scheduler)
        self.save_metrics()

    def save_model(self, model=None, model_path: str = None):
        if model is not None:
            torch.save(self.config.model.state_dict(), os.path.join(self.output_dir, "model.pth") if model_path is None else model_path)

    def save_optimizer(self, optimizer=None, optimizer_path: str = None):
        if optimizer is not None:
            torch.save(self.config.optimizer.state_dict(), os.path.join(self.output_dir, "optimizer.pth") if optimizer_path is None else optimizer_path)

    def save_scheduler(self, scheduler=None, scheduler_path: str = None):
        if scheduler is not None:
            torch.save(self.config.scheduler.state_dict(), os.path.join(self.output_dir, "scheduler.pth") if scheduler_path is None else scheduler_path)

    def save_metrics(self, metrics_path: str = None):
        if metrics_path is not None:
            torch.save(self.metrics, os.path.join(self.output_dir, "metrics.pth") if metrics_path is None else metrics_path)

    def load(self, checkpoint_dir: str = None):
        assert checkpoint_dir is not None, f"checkpoint dir is required"
        self.config.model = self.load_model(self.config.model)
        self.config.optimizer = self.load_optimizer(self.config.optimizer, os.path.join(checkpoint_dir, "optimizer.pth"))
        self.config.scheduler = self.load_scheduler(self.config.scheduler, os.path.join(checkpoint_dir, "scheduler.pth"))
        metrics = self.load_metrics(os.path.join(checkpoint_dir, "metrics.pth"))
        self.__dict__.update(metrics)
        return self.config

    def load_model(self, model=None, model_path: str = None):
        if model is None or model_path is None:
            raise ValueError("model architecture or path must be provided")
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=model.device), strict=False)
        return model

    def load_optimizer(self, optimizer=None, optimizer_path: str = None):
        if optimizer is None or optimizer_path is None:
            raise ValueError("optimizer or path must be provided")
        optimizer.load_state_dict(torch.load(optimizer_path))
        return optimizer

    def load_scheduler(self, scheduler=None, scheduler_path: str = None):
        if scheduler is None or scheduler_path is None:
            raise ValueError("scheduler or path must be provided")
        scheduler.load_state_dict(torch.load(scheduler_path))
        return scheduler

    def load_metrics(self, metrics_path: str = None):
        if metrics_path is None:
            raise ValueError("metrics path must be provided")
        metric = torch.load(metrics_path)
        return metric


class HFCheckpoint(Checkpoint):
    def __init__(self, config: CheckpointConfig, output_dir: str = None):
        super().__init__(config, output_dir)

    def _prepare(self):
        super()._prepare()
        if hasattr(self.config.model, "module"):
            self.config.model = self.config.model.module

    def save(self, **kwargs) -> None:
        self.save_model(self.config.model)
        self.save_optimizer(self.config.optimizer)
        self.save_scheduler(self.config.scheduler)
        self.save_metrics()
        self.save_model_config(self.config.model)
        self.save_tokenizer(kwargs.get("tokenizer", None))

    def save_model(self, model=None, model_path: str = None):
        if model is not None:
            torch.save(model.state_dict(), os.path.join(self.output_dir, "model.bin") if model_path is None else model_path)

    def save_model_config(self, model=None, model_config_path: str = None):
        if model is not None:
            try:
                model.config.to_json_file(os.path.join(self.output_dir, "config.json") if model_config_path is None else model_config_path)
            except Exception as e:
                print("config can't be saved")

    def save_tokenizer(self, tokenizer=None):
        if tokenizer is not None:
            tokenizer.save_pretrained(self.output_dir)

    # def load(self, checkpoint_dir: str = None):
    #     super().load(checkpoint_dir)

