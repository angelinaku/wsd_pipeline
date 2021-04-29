import importlib
import os
import random
import shutil
from typing import Any

import hydra
import numpy as np
import torch


def load_obj(obj_path: str, default_obj_path: str = '') -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)


def set_seed(seed: int = 666) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model_code(model_name) -> None:
    model_path = "models\\" + model_name
    print(model_path)
    print(hydra.utils.get_original_cwd())
    print(os.getcwd())

    os.makedirs(os.path.join(os.getcwd(), 'code'))

    hydra_dir_path = hydra.utils.get_original_cwd()
    save_folder_path = os.getcwd()

    shutil.copy(
        os.path.join(hydra_dir_path, model_path),
        os.path.join(save_folder_path, 'code'),
    )
    shutil.copy2(
        os.path.join(hydra.utils.get_original_cwd(), 'train.py'),
        os.path.join(os.getcwd(), 'code'),
    )
