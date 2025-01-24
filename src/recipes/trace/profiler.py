import contextlib

import torch


@contextlib.contextmanager
def profile(config):
    max_steps = config.stop_steps
    # profiler needs a warmup stage to get the accurate profiling results
    wait_step, warmup_step, active_step = 1, 2, 3
    min_step = wait_step + warmup_step + active_step + 1
    output_dir = f"{config.output_dir}/profiler"
    if 0 < max_steps < min_step:
        raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {max_steps}")
    print(f"pytorch profiling is activated and results will be saved in {output_dir}")
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                output_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
    ) as torch_profiler:
        yield torch_profiler
