from dataclasses import dataclass

import numpy as np


@dataclass
class ImageFile:
    data: np.ndarray
    basename: str
