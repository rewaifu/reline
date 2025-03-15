from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ImageFile:
    data: np.ndarray
    basename: str
    dir: Optional[str] = None
