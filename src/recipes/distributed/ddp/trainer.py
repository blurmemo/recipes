from contextlib import nullcontext
import os
import json
from datetime import datetime

import torch
import torch.cuda.amp
from tqdm import tqdm
from recipes.models.utils import to_device
from recipes.checkpoints.config import CheckpointConfig
from recipes.checkpoints.checkpoint import Checkpoint

from recipes.distributed.utils import get_all_reduce_mean, barrier

class Trainer:
    def __init__(
            self, train_config, model,
            train_dataloader, eval_dataloader,
            data_processor, tokenizer,
            optimizer, scheduler,
            local_rank=None, rank=None, world_size=None,
            **kwargs
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
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
        self.perplexity = 0.0
        self.best_loss = float('inf')
        self.train_step_loss = []
        self.train_step_perplexity = []
        self.pbar = None
        # eval setting
        self.eval_step = 0
        self.eval_max_steps = 0
        self.eval_stop_steps = self.train_config.eval_stop_steps
        self.eval_loss = 0.0
        self.eval_perplexity = 0.0
        self.eval_step_loss = []
        self.eval_step_perplexity = []
        self.eval_pbar = None

        # dist
        self.reduce_loss = []
        self.reduce_perplexity = []


    def save_checkpoint(self):
        config = CheckpointConfig(step=self.step, batch_size=self.train_config.batch_size, loss=self.loss, best_loss=self.best_loss)
        checkpoint = Checkpoint(config, output_dir=self.train_config.output_dir)
        checkpoint.save()

    def before_train(self):
        self.max_steps = len(self.train_dataloader) * self.train_config.epoch // self.train_config.gradient_accumulation_steps
        self.pbar = tqdm(colour="blue", desc=f"Train Step: {self.step}", total=self.max_steps, dynamic_ncols=True)
        self.model.train()
        self.autocast = torch.cuda.amp.autocast if self.train_config.amp else nullcontext
        self.scaler = torch.cuda.amp.GradScaler()
        os.makedirs(self.train_config.output_dir, exist_ok=True)


    def after_train(self):
        if self.rank == 0:
            self.save_checkpoint()
        # save ...
        metrics_filename = f"{self.train_config.output_dir}/train_metrics-{self.rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        save_metrics(metrics_filename, self.step, self.loss, self.perplexity, self.train_step_loss, self.train_step_perplexity)
        barrier()


    def train(self):
        self.before_train()
        for step, batch in self.train_dataloader.sample():
            self.step = step
            if self.step < self.stop_steps: break
            self.before_step()
            loss = self.run_step(step, batch)
            self.backward(loss)
            self.after_step(loss)
            self.reduce()
            self.pbar.update(1)
            self.pbar.set_description(f"Train Step: {self.step}/{self.max_steps} complete (loss: )")
        self.pbar.close()
        self.after_train()

    def before_step(self):
        if self.step % self.train_config.val_interval == 0:
            self.evaluate()


    def after_step(self, loss):
        self.train_step_loss.append(loss.detach().float().item())
        self.perplexity = torch.exp(loss.detach().float())
        self.train_step_perplexity.append(float(torch.exp(loss.detach().float())))

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
            self.reduce_loss.append(float(reduce_loss))
            self.reduce_perplexity.append(float(torch.exp(reduce_loss)))
            self.scheduler.step()


    def backward(self, loss):
        if self.train_config.model_type is torch.float16 and self.train_config.amp:
            self.scaler.scale(loss).backward()
            self.scale_gradient_step()
        else:
            loss.backward()
            self.gradient_step()


    def gradient_step(self):
        if (self.step + 1) % self.train_config.gradient_accumulation_steps == 0 or self.step == self.max_steps - 1:
            if self.train_config.gradient_clip and self.train_config.gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

    def scale_gradient_step(self):
        if (self.step + 1) % self.train_config.gradient_accumulation_steps == 0 or self.step == self.max_steps - 1:
            if self.train_config.gradient_clip and self.train_config.gradient_clip_norm > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.gradient_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()


    def before_evaluate(self):
        self.eval_step = 0
        self.eval_loss = 0.0
        self.eval_perplexity = 0.0
        self.eval_max_steps = len(self.eval_dataloader)
        self.eval_step_loss.clear()
        self.eval_step_perplexity.clear()
        self.pbar = tqdm(colour="green", desc=f"Evaluate Step: {self.eval_step}", total=self.eval_max_steps, dynamic_ncols=True)
        self.model.eval()

    def after_evaluate(self):
        self.eval_loss /= self.eval_step
        self.eval_perplexity = torch.exp(self.eval_loss)
        if self.best_loss > self.eval_loss:
            self.best_loss = self.eval_loss
        # save ...
        metrics_filename = f"{self.train_config.output_dir}/evaluate_metrics-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        save_metrics(metrics_filename, self.eval_step, self.eval_loss, self.eval_perplexity, self.eval_step_loss, self.eval_step_perplexity)



    def evaluate(self):
        self.before_evaluate()
        for step, batch in self.train_dataloader.sample():
            self.eval_step = step
            if self.eval_step < self.eval_stop_steps: break
            self.evaluate_step(step, batch)
            self.pbar.update(1)
            self.pbar.set_description(f"Evaluate Step: {self.eval_step}/{self.eval_max_steps} complete (loss: )")
        self.pbar.close()
        self.after_evaluate()

    def evaluate_step(self, step, batch):
        batch = to_device(batch, self.device)
        with torch.no_grad():
            output = self.model(**batch)
            loss = output.loss
            self.eval_loss += loss.detach().float()
        # preds = torch.argmax(output.logits, dim=-1)
        self.eval_step_loss.append(loss.detach().float().item())
        self.eval_step_perplexity.append(float(torch.exp(loss.detach().float())))


def save_metrics(output_filename, step, loss, perplexity, step_loss, step_perplexity):
    metrics = {
        "step": step,
        "loss": loss,
        "perplexity": perplexity,
        "step_loss": step_loss,
        "step_perplexity": step_perplexity,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics, f)