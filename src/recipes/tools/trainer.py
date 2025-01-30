from contextlib import nullcontext
import os
import time
import json
from datetime import datetime
import numpy as np

import torch
import torch.cuda.amp
from tqdm import tqdm
from recipes.models.utils import to_device
from recipes.checkpoints.config import CheckpointConfig
from recipes.checkpoints.checkpoint import Checkpoint
from recipes.trace.profiler import profile
from recipes.trace.memory import MemoryTrace
from recipes.distributed.utils import get_all_reduce_mean, barrier

class Trainer:
    def __init__(
            self, train_config, model,
            train_dataloader, eval_dataloader,
            data_processor, tokenizer,
            optimizer, scheduler,
            wandb_run=None, **kwargs
    ):
        self.train_config = train_config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = model.device
        self.autocast = None
        self.scaler = None
        self.step = 0
        self.max_steps = 0
        self.stop_steps = self.train_config.stop_steps
        self.loss = 0.0
        self.loss_eval = 0.0
        self.best_loss = float('inf')
        self.train_loss = []
        self.train_perplexity = []
        self.train_step_loss = []
        self.train_step_perplexity = []
        self.pbar = None
        # eval setting
        self.eval_step = 0
        self.eval_max_steps = 0
        self.eval_stop_steps = self.train_config.eval_stop_steps
        self.eval_loss = []
        self.eval_perplexity = []
        self.eval_step_loss = []
        self.eval_step_perplexity = []
        self.eval_pred = []
        self.eval_label = []
        self.eval_pbar = None

        self.checkpoint_time = None

        self.wandb_run = wandb_run


    def save_checkpoint(self):
        start_time = time.perf_counter()
        config = CheckpointConfig(
            step=self.step, batch_size=self.train_config.train_batch_size, loss=self.loss, best_loss=self.best_loss,
            model=self.model, optimizer=self.optimizer, scheduler=self.scheduler
        )
        output_dir = os.path.join(self.train_config.output_dir, f"checkpoint-{self.step}")
        checkpoint = Checkpoint(config, output_dir=output_dir)
        checkpoint.save()
        end_time = time.perf_counter()
        self.checkpoint_time = end_time - start_time


    def wandb_log(self, partition: str="train", **kwargs):
        if self.wandb_run:
            data = {f"{partition}/{k}": v for k, v in kwargs.items()}
            self.wandb_run.log(data)

    def before_train(self):
        self.max_steps = len(self.train_dataloader) * self.train_config.epoch
        self.pbar = tqdm(colour="blue", desc=f"Train Step: {self.step+1}", total=self.max_steps, dynamic_ncols=True)
        self.model.train()
        self.autocast = torch.cuda.amp.autocast if self.train_config.amp else nullcontext
        self.scaler = torch.cuda.amp.GradScaler()
        os.makedirs(self.train_config.output_dir, exist_ok=True)


    def after_train(self):
        # save ...
        metrics_filename = f"{self.train_config.output_dir}/train_metrics-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
        save_metrics(metrics_filename, self.step, self.train_loss, self.train_perplexity, self.train_step_loss, self.train_step_perplexity)
        # result
        result = {}
        result["average_train_loss"] = sum(self.train_loss) / len(self.train_loss)
        result["average_train_perplexity"] = sum(self.train_perplexity) / len(self.train_perplexity)
        result["average_validation_loss"] = sum(self.eval_loss) / len(self.eval_loss)
        result["average_validation_perplexity"] = sum(self.eval_perplexity) / len(self.eval_perplexity)
        result["output_dir"] = self.train_config.output_dir
        result["checkpoint_time"] = self.checkpoint_time
        return result


    def train(self):
        self.before_train()
        with MemoryTrace() as self.memtrace, profile(self.train_config) as self.profiler:
            for step, batch in self.train_dataloader.sample():
                self.step = step
                if self.step == self.stop_steps: break
                self.before_step()
                loss = self.run_step(step, batch)
                self.backward(loss)
                self.after_step(loss)
                self.reduce()
                self.trace()
            self.pbar.close()
        result = self.after_train()
        return result

    def before_step(self):
        if self.step % self.train_config.eval_interval == 0 or self.step == self.stop_steps-1:
            self.evaluate()

    def after_step(self, loss):
        self.train_step_loss.append(loss.detach().float().item())
        perplexity = torch.exp(loss.detach().float())
        self.train_step_perplexity.append(float(torch.exp(loss.detach().float())))
        self.wandb_log("train", step=self.step, loss=loss.detach().float(), perplexity=perplexity)
        self.pbar.update(1)
        self.pbar.set_description(f"Train Step: {self.step+1}/{self.max_steps} complete (loss: {loss.detach().float()})")

    def run_step(self, step, batch):
        batch = to_device(batch, self.device)
        with self.autocast():
            loss = self.model(**batch).loss
        self.loss += loss.detach().float()
        loss = loss / self.train_config.gradient_accumulation_steps
        return loss

    def reduce(self):
        if self.step % len(self.train_dataloader) == 0:
            reduce_loss = get_all_reduce_mean(self.loss) / len(self.train_dataloader)
            self.train_loss.append(float(reduce_loss))
            self.train_perplexity.append(float(torch.exp(reduce_loss)))
            self.loss = 0.0
            self.scheduler.step()

    def trace(self):
        self.profiler.step()
        if (self.step + 1) % len(self.train_dataloader) == 0:
            self.memtrace.print_stats()


    def backward(self, loss):
        if self.train_config.model_dtype is torch.float16 and self.train_config.amp:
            self.scaler.scale(loss).backward()
            self.scale_gradient_step()
        else:
            loss.backward()
            self.gradient_step()


    def gradient_step(self):
        if (self.step + 1) % self.train_config.gradient_accumulation_steps == 0 or self.step == self.max_steps - 1:
            if self.train_config.gradient_clip and self.train_config.gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.gradient_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()


    def scale_gradient_step(self):
        if (self.step + 1) % self.train_config.gradient_accumulation_steps == 0 or self.step == self.max_steps - 1:
            if self.train_config.gradient_clip and self.train_config.gradient_clip_norm > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()


    def before_evaluate(self):
        self.eval_step = 0
        self.loss_eval = 0.0
        self.eval_step_loss.clear()
        self.eval_step_perplexity.clear()
        self.eval_pred.clear()
        self.eval_label.clear()
        self.eval_max_steps = len(self.eval_dataloader)
        self.eval_pbar = tqdm(colour="green", desc=f"Evaluate Step: {self.eval_step+1}", total=self.eval_max_steps, dynamic_ncols=True)
        self.model.eval()

    def after_evaluate(self):
        self.loss_eval /= self.eval_step
        perplexity_eval = torch.exp(self.loss_eval)
        self.eval_loss.append(self.loss_eval)
        self.eval_perplexity.append(perplexity_eval)
        if self.best_loss > self.loss_eval:
            self.best_loss = self.loss_eval
            self.save_checkpoint()
            self.wandb_log("checkpoint", step=self.eval_step, best_loss=self.best_loss, checkpoint_time=self.checkpoint_time)
        # save ...
        metrics_filename = f"{self.train_config.output_dir}/evaluate_metrics-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
        save_metrics(metrics_filename, self.eval_step, self.loss_eval, perplexity_eval, self.eval_step_loss, self.eval_step_perplexity)
        outputs_filename = f"{self.train_config.output_dir}/evaluate_values-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
        save(outputs_filename, prediction=self.eval_pred, label=self.eval_label, tokenizer=self.tokenizer)
        barrier()


    def evaluate(self):
        self.before_evaluate()
        for step, batch in self.train_dataloader.sample():
            self.eval_step = step
            if self.eval_step == self.eval_stop_steps: break
            loss = self.evaluate_step(step, batch)
            self.eval_pbar.update(1)
            self.eval_pbar.set_description(f"Evaluate Step: {self.eval_step+1}/{self.eval_max_steps} complete (loss: {loss.detach().float()})")
        self.eval_pbar.close()
        self.after_evaluate()

    def evaluate_step(self, step, batch):
        batch = to_device(batch, self.device)
        with torch.no_grad():
            output = self.model(**batch)
            loss = output.loss
            self.loss_eval += loss.detach().float()
        preds = torch.argmax(output.logits, dim=-1)
        labels = batch["labels"]
        self.eval_pred.extend(preds.cpu())
        self.eval_label.extend(labels.cpu())
        perplexity = float(torch.exp(loss.detach().float()))
        self.eval_step_loss.append(loss.detach().float().item())
        self.eval_step_perplexity.append(perplexity)
        self.wandb_log("validation", step=self.eval_step, loss=loss.detach().float(), perplexity=perplexity)
        return loss


def save_metrics(output_filename, step, loss, perplexity, step_loss, step_perplexity):
    metrics = {
        "step": step,
        "loss": loss,
        "perplexity": perplexity,
        "step_loss": step_loss,
        "step_perplexity": step_perplexity,
    }
    # with open(output_filename, "w") as f:
    #     json.dump(metrics, f)
    torch.save(metrics, output_filename)

def save(output_filename, **kwargs):
    torch.save(kwargs, output_filename)