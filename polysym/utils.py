import logging
import random
import numpy as np
import torch


class _RandConst:
    """Pickleable constant‐generator with per‐instance bounds."""
    def __init__(self, min_c: float, max_c: float):
        self.min_c = min_c
        self.max_c = max_c
    def __call__(self) -> float:
        return random.uniform(self.min_c, self.max_c)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_logger(name: str = "polysym", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
