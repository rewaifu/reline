from dataclasses import dataclass

import cv2
import numpy as np
from pepe_hyst import monochrome

from reline.static import ImageFile, Node, NodeOptions


@dataclass(frozen=True)
class HystNormOptions(NodeOptions):
    blur_n: int | None = 3
    window_radius: int | None = 3
    min_prominence: float | None = 0.5
    min_distance: int | None = 10
    percentage: float | None = 0.22


def to_u8(img: np.ndarray) -> np.ndarray:
    img = img.squeeze()
    shape = img.shape
    if len(shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return (img * 255.0).clip(0, 255.0).astype(np.uint8)


def to_f32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


class HystNormNode(Node[HystNormOptions]):
    def __init__(self, options):
        super().__init__(options)

    def normalize(self, img: np.ndarray) -> np.ndarray:
        return to_f32(
            monochrome(
                to_u8(img),
                self.options.blur_n,
                self.options.window_radius,
                self.options.min_prominence,
                self.options.min_distance,
                self.options.percentage,
            )
        )

    def process(self, files: list[ImageFile]) -> list[ImageFile]:
        for file in files:
            file.data = self.normalize(
                file.data,
            )

        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        file.data = self.normalize(
            file.data,
        )
        return file

    def video_process(self, file: np.ndarray) -> np.ndarray:
        file = self.normalize(
            file,
        )
        return file
