import math
import torch
from torch import nn
import torch.nn.functional as F


class LinearLoraLayer(nn.Module):
    def __init__(self, weight, r=0, alpha=1, dropout=0, bias=None):
        super(LinearLoraLayer, self).__init__()
        self.weight = weight
        self.bias = bias

        if r <= 0:
            raise ValueError("You are training to use LoRA, whose reduced dim should be larger than 1")

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(columns, r))
        self.lora_left_weight = nn.Parameter(torch.zeros(r, rows))
        self.lora_scaling = alpha / r

        if dropout > 0:
            self.lora_dropout = nn.Dropout(dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()
        
    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, x):
        if self.fuse_lora:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias) + (self.lora_dropout(x) @ self.lora_right_weight @ self.lora_left_weight) * self.lora_scaling

