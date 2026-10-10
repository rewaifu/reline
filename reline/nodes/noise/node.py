from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import cv2
import numpy as np
from numpy.random import Generator, default_rng

from reline.static import ImageFile, Node, NodeOptions


class NoiseMode(Enum):
    Gray = 1
    RGB = 2
    Chrome = 3


NoiseMap = {'gray': NoiseMode.Gray, 'rgb': NoiseMode.RGB, 'chrome': NoiseMode.Chrome}
NoiseType = Literal['gray', 'rgb', 'chrome']


def generate_beta_noise(shape: Sequence[int], a: float, b: float, rng: Generator):
    return (rng.beta(a, b, shape) * 2 - 1).astype(np.float32)


@dataclass(frozen=True)
class NoiseOptions(NodeOptions):
    a: float = 1
    b: float = 1
    alpha: float = 0.1
    noise_mode: NoiseType = 'rgb'
    th_min: int = 1
    th_max: int = 254
    seed: int | None = None


class NoiseNode(Node[NoiseOptions]):
    def __init__(self, options):
        super().__init__(options)
        self.noise_mode = NoiseMap[self.options.noise_mode]

    def add_noise(
        self,
        img: np.ndarray,
        rng: Generator,
    ):
        noise_type = self.noise_mode
        img = img.squeeze()
        mask = (img <= self.options.th_min/255.0) | (img >= self.options.th_max/255.0)
        short_img = img[mask]
        if len(img.shape) == 2:
            noise_type = NoiseMode.RGB
        if noise_type == NoiseMode.Gray:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            max_lab = np.max(lab, (0, 1))
            min_lab = np.min(lab, (0, 1))
            lab = (lab - min_lab[None, None, :]) / (max_lab - min_lab)[None, None, :]
            lab[:, :, 0] = (lab[:, :, 0] + generate_beta_noise(img.shape[:2], self.options.a, self.options.b, rng) * self.options.alpha).clip(0, 1)
            lab = lab * (max_lab - min_lab)[None, None, :] + min_lab[None, None, :]
            img = cv2.cvtColor(lab, cv2.COLOR_LUV2RGB)
        elif noise_type == NoiseMode.Chrome:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            max_lab = np.max(lab, (0, 1))
            min_lab = np.min(lab, (0, 1))
            lab = (lab - min_lab[None, None, :]) / (max_lab - min_lab)[None, None, :]
            lab[:, :, 1:] = (lab[:, :, 1:] + generate_beta_noise([*img.shape[:2], 2], self.options.a, self.options.b, rng) * self.options.alpha).clip(
                0, 1
            )
            lab = lab * (max_lab - min_lab)[None, None, :] + min_lab[None, None, :]
            img = cv2.cvtColor(lab, cv2.COLOR_LUV2RGB)
        else:
            img = (img + generate_beta_noise(img.shape, self.options.a, self.options.b, rng) * self.options.alpha).clip(0, 1)
        img[mask]= short_img
        return img

    def process(self, files: list[ImageFile]) -> list[ImageFile]:
        for file in files:
            file.data = self.add_noise(file.data, default_rng(self.options.seed))

        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        file.data = self.add_noise(file.data, default_rng(self.options.seed))
        return file

    def video_process(self, file: np.ndarray) -> np.ndarray:
        file = self.add_noise(file, default_rng(self.options.seed))
        return file
