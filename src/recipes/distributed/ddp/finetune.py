from recipes.distributed.ddp.training import TrainConfig as DDP_TRAINING_CONFIG
from recipes.utils.utils import set_seed
from recipes.distributed.utils import init_distributed, get_dist_info, setup_gpu
from recipes.data.processor import Processor
from recipes.models.model import Model
from recipes.distributed.ddp.wrapper import Wrapper as DDPWrapper

from recipes.data.dataloader import DistributedDataLoader

from recipes.optimizers.config import OptimizerConfig
from recipes.optimizers.optimizer import Optimizer

from recipes.schedulers.config import SchedulerConfig
from recipes.schedulers.scheduler import Scheduler

from recipes.distributed.ddp.trainer import Trainer as DDPTrainer

def main():
    train_config = DDP_TRAINING_CONFIG()

    set_seed(train_config.seed)

    init_distributed()

    local_rank, rank, world_size = get_dist_info()

    setup_gpu(local_rank, rank)

    data_processor, tokenizer = Processor.build(train_config)

    model = Model(train_config, tokenizer).arch
    model.to(rank)

    model = DDPWrapper(model, rank).arch

    train_dataloader = DistributedDataLoader(train_config, data_processor, train_config.train_batch_size, "train", rank=rank, world_size=world_size)

    eval_dataloader = DistributedDataLoader(train_config, data_processor, train_config.val_batch_size, "val", rank=rank, world_size=world_size)

    optimizer_config = OptimizerConfig.generate(train_config)
    optimizer = Optimizer.build(optimizer_config, model)

    scheduler_config = SchedulerConfig.generate(train_config)
    scheduler = Scheduler.build(scheduler_config, optimizer)

    ddp_trainer = DDPTrainer(
        train_config,
        model,
        train_dataloader,
        eval_dataloader,
        data_processor,
        tokenizer,
        optimizer,
        scheduler,
    )
    ddp_trainer.train()


if __name__ == '__main__':
    main()