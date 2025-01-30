from recipes.configs.training import TrainConfig as TRAINING_CONFIG
from recipes.configs.wandb import WandbConfig as WANDB_CONFIG
from recipes.utils.utils import set_seed
from recipes.distributed.utils import init_distributed, get_dist_info, setup_gpu, cleanup
from recipes.models.model import Model
from recipes.data.processor import Processor
from recipes.utils.utils import compute_model_size, print_model_status
from recipes.tuners.lora.config import LoraConfig
from recipes.tuners.tuner import Tuner
from recipes.models.utils import freeze


from recipes.optimizers.config import OptimizerConfig
from recipes.optimizers.optimizer import Optimizer

from recipes.schedulers.config import SchedulerConfig
from recipes.schedulers.scheduler import Scheduler

from recipes.distributed.ddp.wrapper import Wrapper as DDPWrapper
from recipes.trace.wandb import Wandb

from recipes.tools.trainer import Trainer


def main():
    train_config, wandb_config = TRAINING_CONFIG(), WANDB_CONFIG()

    set_seed(train_config.seed)

    init_distributed()

    local_rank, rank, world_size = get_dist_info()

    setup_gpu(local_rank, rank)


    data_processor, tokenizer = Processor.build(train_config)

    model = Model(train_config, tokenizer).arch

    model_size = compute_model_size(model)

    print(f"\n--> model has {model_size / 1e6} million parameters.\n")

    # freeze
    model = freeze(model)

    # peft
    lora_config = LoraConfig()
    model.language_model = Tuner(lora_config, model.language_model).arch

    peft_model_size = compute_model_size(model)

    print(f"\n--> model has {peft_model_size / 1e6} million trainable parameters.\n")

    print_model_status(model)

    wandb_run = Wandb.generate(wandb_config, [train_config, lora_config], output_dir=train_config.output_dir) if rank == 0 else None

    model.to("cuda")

    model = DDPWrapper(model, rank=rank).arch

    # train_dataloader = DistributedDataLoader(train_config, data_processor, tokenizer, train_config.train_batch_size, "train", rank=rank, world_size=world_size)
    # eval_dataloader = DistributedDataLoader(train_config, data_processor, tokenizer, train_config.train_batch_size, "val", rank=rank, world_size=world_size)
    train_dataloader = None
    eval_dataloader = None

    optimizer_config = OptimizerConfig.generate(train_config)
    optimizer = Optimizer.build(optimizer_config, model)

    scheduler_config = SchedulerConfig.generate(train_config)
    scheduler = Scheduler.build(scheduler_config, optimizer)

    trainer = Trainer(
        train_config,
        model,
        train_dataloader,
        eval_dataloader,
        data_processor,
        tokenizer,
        optimizer,
        scheduler,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        wandb_run=wandb_run,
    )

    result = trainer.train()

    if rank == 0:
        for k, v in result.items():
            print(f"{k}:  {v}")
            wandb_run.summary[k] = v

    cleanup()


if __name__ == '__main__':
    main()