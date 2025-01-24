import contextlib

import torch

from recipes.trace.flop import FlopMeasure

@contextlib.contextmanager
def profile(config, local_rank: int=None):
    mode = config.profile_mode
    max_steps = config.stop_steps
    if mode == "profile":
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if 0 < max_steps < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {max_steps}")
        print(f"pytorch profiling is activated and results will be saved in {config.output_dir}")
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    config.output_dir
                ),
                profile_memory=True,
                with_stack=False,
                with_flops=True,
                record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif mode == "flop":
        if 0 < max_steps <= config.flop_counter_start:
            raise ValueError(f"flop counter requires at least {config.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {max_steps}")
        with FlopMeasure(rank=local_rank, warmup_step=config.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None