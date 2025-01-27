from .PointGPT import PointTransformer
from .config import get_config
from typing import Tuple

def load_pointgpt(config_path:str, checkpoint_path:str) -> Tuple[PointTransformer, float]:
    config_data = get_config(config_path)
    model_config = config_data['model']
    model = PointTransformer(model_config).to("cuda:0")
    model.load_model_from_ckpt(checkpoint_path)
    max_depth = config_data['max_depth']
    return model, max_depth

