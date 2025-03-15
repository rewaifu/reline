from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

import numpy as np

from .file import ImageFile


class NodeOptions(ABC):
    pass


T = TypeVar('T', bound=NodeOptions, covariant=True)


class Node(ABC, Generic[T]):
    def __init__(self, options: T):
        self.options = options

    def update_options(self, options: T):
        self.options = options

    @abstractmethod
    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        raise NotImplementedError

    @abstractmethod
    def single_process(self, file: ImageFile) -> ImageFile:
        raise NotImplementedError

    @abstractmethod
    def video_process(self, file: np.ndarray) -> np.ndarray:
        raise NotImplementedError
