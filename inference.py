import os
import warnings
import json
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from utils.utils import set_seed, load_obj

warnings.filterwarnings('ignore')


def run(conf: DictConfig, current_dir) -> None:
    """
    Run pytorch-lightning model

    Args:
        new_dir:
        conf: hydra config

    """
    set_seed(conf.training.random_seed)
    hparams = OmegaConf.to_container(conf)

    trainer = pl.Trainer(**conf.trainer)

    dm = load_obj(conf.training.data_module_name)(hparams=hparams, conf=conf)
    dm.setup()

    model = load_obj(conf.training.lightning_module_name)(hparams=hparams, conf=conf, tag_to_idx=dm.tag_to_idx,
                                                          embedder=dm.embedder)

    # best_path = 'C:/Users/Ангелина/Python_codes/wsd_train_folder/outputs/2021-02-02_16-54-30/saved_models/epoch=22_valid_score_mean=0.9609.ckpt'
    best_path = 'C:/Users/Ангелина/Python_codes/wsd_train_folder/outputs/2021-02-09_19-27-50_elmo/saved_models/epoch=22_valid_score_mean=0.9617.ckpt'
    model = model.load_from_checkpoint(
        best_path, hparams=hparams, conf=conf, tag_to_idx=dm.tag_to_idx, embedder=dm.embedder, strict=False
    )
    save_name = best_path.split('/')[-1][:-5]
    model_name = f'C:/Users/Ангелина/Python_codes/wsd_train_folder/outputs/2021-02-09_19-27-50_elmo/saved_models/{save_name}.pth'
    print(model_name)
    torch.save(model.wsd_model.state_dict(), model_name)
    #
    # output_dict = trainer.test(model=model, datamodule=dm)
    #
    # with open(os.path.join(current_dir, 'test_output.json'), 'w') as f:
    #     json.dump(list(output_dict[0]['predicted']), f, ensure_ascii=False, indent=4)

# @hydra.main(config_path=r'C:\Users\Ангелина\Python_codes\wsd_train_folder\outputs\2021-02-02_16-54-30\logs\csv_experiment_log\version_0\\', config_name='hparams_2')
@hydra.main(config_path=r'C:\Users\Ангелина\Python_codes\wsd_train_folder\outputs\2021-02-09_19-27-50_elmo\logs\csv_experiment_log\version_0\\', config_name='hparams_2')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    print(cfg.pretty())
    run(cfg, os.getcwd())


if __name__ == '__main__':
    run_model()