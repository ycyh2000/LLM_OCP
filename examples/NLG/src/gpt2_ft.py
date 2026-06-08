#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import numpy as np
import itertools

import torch
import random
from torch.utils.data import DataLoader

torch.set_printoptions(threshold=100000)

from gpu import (
    add_gpu_params,
    parse_gpu,
    distributed_opt,
    distributed_gather,
    distributed_sync,
    cleanup
)
# from optimizer import (
#     create_adam_optimizer,
#     create_optimizer_scheduler,
#     add_optimizer_params,
#     create_adam_optimizer_from_args
# )

from optimizer import *
# from galore_optimizer import *

from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

import loralib as lora

parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', required=True, help='location of training data corpus')

parser.add_argument('--valid_data', required=True, help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumulation steps')

parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'],
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')

parser.add_argument('--fp16', action='store_true', help='train model with fp16')

parser.add_argument('--log_interval', type=int, default=100, help='log interval')

parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')

parser.add_argument('--save_interval', type=int, default=500, help='save interval')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'),
                    help='working folder.')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'],
                    help='language model training objective')

parser.add_argument('--lora_dropout', default=0.0, type=float,
                    help='dropout probability for lora layers')

parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')

parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval')

parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate')

parser.add_argument('--roll_step', type=int, default=100, help='rolling step')

parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs')

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import os
from collections import OrderedDict
import argparse

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from galore_projector import GaLoreProjector
from galore_projector_tensor import GaLoreProjectorTensor


def add_optimizer_params(parser: argparse.ArgumentParser):
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay rate')
    parser.add_argument('--correct_bias', action='store_true', help='correct adam bias term')
    parser.add_argument('--adam_epislon', default=1e-6, type=float, help='adam epsilon')
    parser.add_argument('--no_decay_bias', action='store_true', help='no weight decay on bias weigh')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='adam beta1 term')
    parser.add_argument('--adam_beta2', default=0.98, type=float, help='adam beta2 term')

    parser.add_argument('--scheduler', default='linear', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'linear', 'cycle', 'None'],
                        help='lr scheduler to use.')

    parser.add_argument('--max_step', type=int, default=None, help='upper epoch limit')

    parser.add_argument('--max_epoch', type=int, default=None, help='max epoch of training')

    parser.add_argument('--warmup_step', type=int, default=0, help='upper epoch limit')

    parser.add_argument('--i_steps', type=str, default='0', help='interval_steps')
    parser.add_argument('--i_lrs', type=str, default='0.00025', help='interval_lrs')

    # GaLore parameters
    parser.add_argument("--galore_rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")


def tensor_mem_mb(t):
    return t.numel() * t.element_size() / 1024 / 1024


def _accumulate_tensor_mem(obj):
    """Return memory in MB for tensors nested in common Python containers."""
    if torch.is_tensor(obj):
        return tensor_mem_mb(obj)
    if isinstance(obj, dict):
        return sum(_accumulate_tensor_mem(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_accumulate_tensor_mem(v) for v in obj)
    return 0.0


def report_memory_breakdown(model, optimizer, tag="", extra_tensors=None, show_summary=False):
    """
    Print a phase-level CUDA memory report.

    The explicit categories below only include tensors reachable from the model,
    optimizer states, GaLore projectors, and optional extra tensors such as the
    current input/target/mask/logits/loss. The remaining CUDA allocated memory is
    mostly activation tensors, autograd saved tensors, temporary workspaces, and
    other framework allocations that cannot be directly reached from model or
    optimizer objects.
    """
    param_mem = 0.0
    trainable_param_mem = 0.0
    frozen_param_mem = 0.0
    grad_mem = 0.0
    opt_state_mem = 0.0
    projector_mem = 0.0
    extra_mem = 0.0

    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        mem = tensor_mem_mb(p)
        param_mem += mem
        total_params += p.numel()
        if p.requires_grad:
            trainable_param_mem += mem
            trainable_params += p.numel()
        else:
            frozen_param_mem += mem
        if p.grad is not None:
            grad_mem += tensor_mem_mb(p.grad)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                opt_state_mem += tensor_mem_mb(v)
            elif k == "projector":
                proj = v
                if hasattr(proj, "ortho_matrix"):
                    mat = proj.ortho_matrix
                    if torch.is_tensor(mat):
                        projector_mem += tensor_mem_mb(mat)
                    elif isinstance(mat, (list, tuple)):
                        for x in mat:
                            if torch.is_tensor(x):
                                projector_mem += tensor_mem_mb(x)

    if extra_tensors is not None:
        extra_mem = _accumulate_tensor_mem(extra_tensors)

    explicit_mem = param_mem + grad_mem + opt_state_mem + projector_mem + extra_mem
    cuda_allocated = torch.cuda.memory_allocated() / 1024 / 1024
    cuda_reserved = torch.cuda.memory_reserved() / 1024 / 1024
    cuda_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    unaccounted_allocated = max(cuda_allocated - explicit_mem, 0.0)

    print("=" * 96)
    print(f"[Memory Breakdown] {tag}")
    print(f"Total params          : {total_params / 1e6:.2f} M")
    print(f"Trainable params      : {trainable_params / 1e6:.2f} M")
    print(f"Parameter memory      : {param_mem:.2f} MB")
    print(f"  - trainable params  : {trainable_param_mem:.2f} MB")
    print(f"  - frozen params     : {frozen_param_mem:.2f} MB")
    print(f"Gradient memory       : {grad_mem:.2f} MB")
    print(f"Optimizer state memory: {opt_state_mem:.2f} MB")
    print(f"GaLore projector mem  : {projector_mem:.2f} MB")
    print(f"Extra tensors memory  : {extra_mem:.2f} MB")
    print(f"Explicit subtotal     : {explicit_mem:.2f} MB")
    print(f"CUDA allocated        : {cuda_allocated:.2f} MB")
    print(f"CUDA reserved         : {cuda_reserved:.2f} MB")
    print(f"CUDA peak allocated   : {cuda_peak:.2f} MB")
    print(f"Unaccounted allocated : {unaccounted_allocated:.2f} MB")
    print(
        "Note: unaccounted memory is typically activations, autograd saved tensors, temporary workspaces, and CUDA/PyTorch runtime allocations.")
    print("=" * 96)

    if show_summary:
        print(torch.cuda.memory_summary())


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_lr: float = 0.1,
            min_lr: float = 0.0,
            warmup_steps: int = 0,
            max_steps: int = 1,
            alpha: float = 0.,
            last_epoch: int = -1
    ):
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size

        self.alpha = alpha  # decrease rate of max learning rate by cycle
        self.max_steps = max_steps
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            curr_lr = self.max_lr * self.last_epoch / self.warmup_steps
            return curr_lr
        else:
            _step = min(self.last_epoch, self.max_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * _step / self.max_steps))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            return self.max_lr * decayed  # learning_rate * decayed

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = _lr


class CyclicScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer,
            interval_steps=[],
            interval_lrs=[],
            last_epoch=-1,
    ):
        self.optimizer = optimizer

        self.interval_steps = interval_steps
        self.interval_lrs = interval_lrs

        self.last_epoch = last_epoch

        super(CyclicScheduler, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.interval_lrs[0]

    def get_lr(self):
        for _i in range(0, len(self.interval_steps) - 1):
            if self.last_epoch >= self.interval_steps[_i] and self.last_epoch < self.interval_steps[_i + 1]:
                _alpha = (self.last_epoch - self.interval_steps[_i]) / (
                        self.interval_steps[_i + 1] - self.interval_steps[_i] + 1e-6)
                if _alpha < 0:
                    _alpha = 0
                if _alpha >= 1:
                    _alpha = 1
                curr_lr = _alpha * self.interval_lrs[_i + 1] + (1.0 - _alpha) * self.interval_lrs[_i]
                return curr_lr
        return self.interval_lrs[-1]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        # self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups:  # , self.get_lr()):
            param_group['lr'] = _lr


def get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_training_steps,
        last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_training_steps,
        last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_grouped_parameters(model, no_decay_bias):  # args):
    if not no_decay_bias:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()],  # if not any(nd in n for nd in no_decay)],
            }]
    else:
        no_decay = ["bias", "layer_norm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }]
    return optimizer_grouped_parameters


def create_adam_optimizer(
        model,
        lr,
        weight_decay,
        optimizer_grouped_parameters=None,
        beta1=0.9,
        beta2=0.98,
        correct_bias=True,
        adam_epislon=1e-6,
        no_decay_bias=False,
        galore_rank=4,
        update_proj_gap=200,
        galore_scale=1.0,
        proj_type='std'
):
    if optimizer_grouped_parameters is None:
        optimizer_grouped_parameters = create_grouped_parameters(model, no_decay_bias)

    optimizer = AdamW_galore(
        optimizer_grouped_parameters,
        lr=lr,
        betas=(beta1, beta2),
        eps=adam_epislon,
        weight_decay=weight_decay,
        correct_bias=correct_bias

        ,
        galore_rank=galore_rank,
        update_proj_gap=update_proj_gap,
        scale=galore_scale,
        proj_type=proj_type
    )

    optimizer.param_name_map = {id(p): name for name, p in model.named_parameters()}


    return optimizer


def create_sgd_optimizer(model, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    return optimizer


def create_adam_optimizer_from_args(model, args, grouped_parameters=None):
    """
    Create an AdamW/GaLore optimizer.

    Important:
    - The GaLore optimizer in this project enables projection through parameter-group
      metadata, such as ``rank``, ``update_proj_gap``, ``scale``, and ``proj_type``.
    - Only matrix-like parameters are assigned to the GaLore group. One-dimensional
      parameters such as bias and LayerNorm weights are kept in a normal AdamW group
      to avoid shape errors in GaLoreProjector.
    """
    if grouped_parameters is None:
        galore_params = []
        regular_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            if p.dim() >= 2:
                galore_params.append(p)
            else:
                regular_params.append(p)

        grouped_parameters = []

        if len(galore_params) > 0:
            grouped_parameters.append({
                "params": galore_params,
                "rank": args.galore_rank,
                "update_proj_gap": args.update_proj_gap,
                "scale": args.galore_scale,
                "proj_type": args.proj_type,
                "dim": 2,
            })

        if len(regular_params) > 0:
            grouped_parameters.append({
                "params": regular_params,
            })

        if args.rank == 0:
            galore_num = sum(p.numel() for p in galore_params)
            regular_num = sum(p.numel() for p in regular_params)
            print(f"GaLore parameter count : {galore_num / 1e6:.2f}M")
            print(f"Regular parameter count: {regular_num / 1e6:.2f}M")

    optimizer = AdamW_galore(
        grouped_parameters,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epislon,
        weight_decay=args.weight_decay,
        correct_bias=args.correct_bias

        ,
        galore_rank=args.galore_rank,
        update_proj_gap=args.update_proj_gap,
        scale=args.galore_scale,
        proj_type=args.proj_type
    )

    optimizer.param_name_map = {id(p): name for name, p in model.named_parameters()}
    # print(f"optimizer.param_name_map = {optimizer.param_name_map}")

    return optimizer


def create_optimizer_scheduler(optimizer, args):
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            max_lr=args.lr,
            min_lr=0.0,
            warmup_steps=args.warmup_step,
            max_steps=args.max_step, alpha=0
        )
    elif args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, args.warmup_step, args.max_step, last_epoch=-1
        )
    elif args.scheduler == 'cycle':
        if args.i_steps is not None:
            args.i_steps = [int(_i) for _i in args.i_steps.split(',')]
            args.i_lrs = [float(_i) for _i in args.i_lrs.split(',')]
        args.max_step = args.i_steps[-1]
        print('max_step is rest to', args.max_step)
        scheduler = CyclicScheduler(
            optimizer, interval_steps=args.i_steps, interval_lrs=args.i_lrs
        )
    elif args.scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, args.warmup_step, args.max_step, last_epoch=-1
        )
    else:
        # constant leanring rate.
        scheduler = None
    return scheduler


# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step(
        _loss,
        _optimizer,
        _model,
        _schedule,
        args,
        is_update=True,
        train_step=None,
        extra_tensors=None,
):
    # should_report = train_step is not None and train_step % 100 == 0 and args.rank == 0
    should_report = False

    if args.fp16:
        with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    # Gradients exist here. This is the right place to measure Gradient memory.
    if should_report:
        report_memory_breakdown(
            _model,
            _optimizer,
            tag=f"after backward step {train_step}",
            extra_tensors=extra_tensors,
        )

    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()

        # Optimizer states are materialized/updated here. Gradients still exist.
        if should_report:
            report_memory_breakdown(
                _model,
                _optimizer,
                tag=f"after optimizer step {train_step}",
                extra_tensors=extra_tensors,
            )

        _optimizer.zero_grad()

        # After zero_grad(), Gradient memory should normally become 0.
        if should_report:
            report_memory_breakdown(
                _model,
                _optimizer,
                tag=f"after zero_grad step {train_step}",
                extra_tensors=extra_tensors,
                show_summary=True,
            )

    if _schedule is not None:
        _schedule.step()


def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}

            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

            _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk)
            loss = _loss.mean()

            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', loss.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        args,
        train_step=0,
        epoch=0
):
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    train_loader.sampler.set_epoch(epoch)

    for idx, data in enumerate(train_loader):
        data = {key: value for key, value in data.items()}

        next_step = train_step + 1
        # should_report = next_step % 100 == 0 and args.rank == 0
        should_report = False
        # Reset peak at the beginning of a reported step so the peak reflects this step
        # rather than the maximum since program start.
        if should_report:
            torch.cuda.reset_peak_memory_stats()
            report_memory_breakdown(
                model,
                optimizer,
                tag=f"before input transfer step {next_step}",
            )

        _input = data['input'].to(args.device)
        _target = data['target'].to(args.device)
        _msk = data['mask'].to(args.device)

        batch_tensors = {
            "input": _input,
            "target": _target,
            "mask": _msk,
        }

        if should_report:
            report_memory_breakdown(
                model,
                optimizer,
                tag=f"after input transfer step {next_step}",
                extra_tensors=batch_tensors,
            )

        _lm_logits, _lm_loss = model(
            _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
        )

        # Include logits and raw loss as explicit tensors at the after-forward point.
        forward_tensors = {
            "input": _input,
            "target": _target,
            "mask": _msk,
            "logits": _lm_logits,
            "loss": _lm_loss,
        }

        if should_report:
            report_memory_breakdown(
                model,
                optimizer,
                tag=f"after forward step {next_step}",
                extra_tensors=forward_tensors,
            )

        _lm_loss = _lm_loss.mean()

        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False
        avg_lm_loss.update(_lm_loss.item())
        optimizer_step(
            _lm_loss / args.grad_acc,
            optimizer,
            model,
            scheduler,
            args,
            is_update=is_update,
            train_step=train_step,
            extra_tensors={
                "input": _input,
                "target": _target,
                "mask": _msk,
                "logits": _lm_logits,
                "loss": _lm_loss,
            },
        )

        if train_step % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | {idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

            if args.rank == 0:
                print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()

        if train_step % args.save_interval == 0:
            if args.rank == 0:
                model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
                print('saving checkpoint', model_path)
                # torch.save({'model_state_dict': lora.lora_state_dict(model)}, model_path)

                if args.lora_dim > 0:
                    save_state_dict = lora.lora_state_dict(model)
                else:
                    save_state_dict = model.state_dict()

                torch.save({'model_state_dict': save_state_dict}, model_path)

            distributed_sync(args)

        # evaluation interval
        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()

            valid_loss, valid_ppl = evaluate(model, valid_loader, args)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl

            log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
                      f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                      f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '

            if args.rank == 0:
                print('-' * 100)
                print(log_str)
                print('-' * 100)

            model.train()
            distributed_sync(args)

        if train_step == args.max_step:
            break

    if args.rank == 0:
        model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path)
    distributed_sync(args)
    return train_step


if __name__ == '__main__':
    args = parser.parse_args()
    parse_gpu(args)
    print_args(args)

    if args.fp16:
        try:
            from apex import amp
        except Exception as e:
            warnings.warn('Could not import amp, apex may not be installed')

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len,
        joint_lm=args.obj == 'jlm'
    )

    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, num_workers=0,
        shuffle=False, pin_memory=False, drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed)
    )

    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0,
        shuffle=False, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed)
    )

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        lm_net.load_weight(torch.load(args.init_checkpoint))

    lm_net = lm_net.cuda()

    # for name, param in lm_net.named_parameters():
    #     print(name, param.shape)

    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net)

    optimizer = create_adam_optimizer_from_args(lm_net, args)

    if args.max_step is None:
        args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        print('set max_step:', args.max_step)

    scheduler = create_optimizer_scheduler(optimizer, args)
    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
    lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                lm_net, optimizer, scheduler, train_loader, valid_loader, args,
                train_step=train_step, epoch=epoch
            )

            if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                if args.rank == 0:
                    print('-' * 100)
                    print('End of training')
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print('-' * 100)
            print('Exiting from training early')

    distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)
