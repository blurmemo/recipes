from recipes.distributed.ds.training import TrainConfig as DS_TRAINING_CONFIG
from recipes.distributed.ds.ds import DSConfig as DS_CONFIG
from recipes.utils.utils import set_seed
from recipes.distributed.utils import init_deepspeed, get_dist_info, setup_gpu
from recipes.data.processor import Processor
from recipes.models.model import Model
from recipes.distributed.ds.wrapper import Wrapper as DSWrapper

from recipes.data.dataloader import DistributedDataLoader

from recipes.optimizers.config import OptimizerConfig
from recipes.optimizers.optimizer import Optimizer

from recipes.schedulers.config import SchedulerConfig
from recipes.schedulers.scheduler import Scheduler


def main():
    train_config, ds_config = DS_TRAINING_CONFIG(), DS_CONFIG

    set_seed(train_config.seed)

    init_deepspeed()

    local_rank, rank, world_size = get_dist_info()

    setup_gpu(local_rank, rank)

    data_processor, tokenizer = Processor.build(train_config)

    model = Model(train_config, tokenizer).arch
    model.to(rank)

    train_dataloader = DistributedDataLoader(train_config, data_processor, train_config.train_batch_size, "train", rank=rank, world_size=world_size)

    eval_dataloader = DistributedDataLoader(train_config, data_processor, train_config.val_batch_size, "val", rank=rank, world_size=world_size)

    optimizer_config = OptimizerConfig.generate(train_config)
    optimizer = Optimizer.build(optimizer_config, model)

    scheduler_config = SchedulerConfig.generate(train_config)
    scheduler = Scheduler.build(scheduler_config, optimizer)

    ds_wrapper = DSWrapper(ds_config, model, optimizer, scheduler, rank=rank)

    model, optimizer, scheduler = ds_wrapper.arch, ds_wrapper.optimizer, ds_wrapper.scheduler

    # trainer = Trainer(...)
    # trainer.train()


if __name__ == '__main__':
    main()