from recipes.configs.training import TrainConfig as TRAINING_CONFIG
from recipes.utils.utils import set_seed
from recipes.data.processor import Processor
from recipes.models.model import Model
from recipes.data.dataloader import DataLoader
from recipes.optimizers.config import OptimizerConfig
from recipes.optimizers.optimizer import Optimizer
from recipes.schedulers.config import SchedulerConfig
from recipes.schedulers.scheduler import Scheduler
from recipes.tools.trainer import Trainer

def main():
    train_config = TRAINING_CONFIG()

    set_seed(train_config.seed)

    data_processor, tokenizer = Processor.build(train_config)

    model = Model(train_config, tokenizer).arch

    model.to(train_config.device_map)

    # train_dataloader = DataLoader(train_config, data_processor, tokenizer, train_config.train_batch_size, "train")
    # eval_dataloader = DataLoader(train_config, data_processor, tokenizer, train_config.train_batch_size, "val")


    optimizer_config = OptimizerConfig.generate(train_config)
    optimizer = Optimizer.build(optimizer_config, model)

    scheduler_config = SchedulerConfig.generate(train_config)
    scheduler = Scheduler.build(scheduler_config, optimizer)

    # trainer = Trainer(...)
    # trainer.train()



if __name__ == '__main__':
    main()