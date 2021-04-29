import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.utils import set_seed, load_obj, save_model_code

warnings.filterwarnings('ignore')


def run(conf: DictConfig) -> None:
    """
    Run pytorch-lightning model

    Args:
        new_dir:
        conf: hydra config

    """
    set_seed(conf.training.random_seed)

    hparams = OmegaConf.to_container(conf)

    # log_save_path = conf.general.all_logs_storage_path

    conf.callbacks.model_checkpoint.params.filepath = os.getcwd() + conf.callbacks.model_checkpoint.params.filepath

    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(**conf.callbacks.model_checkpoint.params)
    early_stop_callback = EarlyStopping(**conf.callbacks.early_stopping.params)

    loggers = []
    if conf.logging.enable_logging:
        for logger in conf.logging.loggers:
            loggers.append(load_obj(logger.class_name)(**logger.params))

    trainer = pl.Trainer(logger=loggers, checkpoint_callback=checkpoint_callback, callbacks=[early_stop_callback],
                         **conf.trainer)

    dm = load_obj(conf.training.data_module_name)(hparams=hparams, conf=conf)
    dm.setup()
    num_steps_in_epoch = len(dm.train_dataloader())

    model = load_obj(conf.training.lightning_module_name)(hparams=hparams, conf=conf, tag_to_idx=dm.tag_to_idx,
                                                          embedder=dm.embedder, num_steps=num_steps_in_epoch)

    trainer.fit(model, dm)

    if conf.general.save_pytorch_model:
        if conf.general.save_best:
            best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            print('Best model score ', trainer.checkpoint_callback.best_model_score)
            # extract file name without folder and extension
            save_name = best_path.split('/')[-1][:-5]
            model = model.load_from_checkpoint(
                best_path, hparams=hparams, conf=conf, tag_to_idx=dm.tag_to_idx, embedder=dm.embedder, strict=False
            )
            model_name = f'saved_models/{save_name}.pth'
            print(model_name)
            torch.save(model.model.state_dict(), model_name)
        else:
            os.makedirs('saved_models', exist_ok=True)
            model_name = 'saved_models/last.pth'
            print(model_name)
            torch.save(model.model.state_dict(), model_name)

    trainer.test(model=model, datamodule=dm)


@hydra.main(config_path='conf', config_name='main_config')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    print(cfg.pretty())
    if cfg.general.log_model_code:
        model_name = str(cfg.model.model_class).split('.')[1] + '.py'
        save_model_code(model_name)
    run(cfg)


if __name__ == '__main__':
    run_model()
