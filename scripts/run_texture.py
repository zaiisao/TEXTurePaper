import pyrallis

from src.configs.train_config import TrainConfig
from src.training.trainer import TEXTure


@pyrallis.wrap()
def main(cfg: TrainConfig):
    #breakpoint()
    trainer = TEXTure(cfg)
    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        #breakpoint()
        # JA: In this code, "paint" function corresponds to the training code
        trainer.paint()


if __name__ == '__main__':
    main()
