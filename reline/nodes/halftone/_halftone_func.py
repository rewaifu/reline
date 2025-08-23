from typing import Literal

import numpy as np
from pepeline import screentone, cvt_color, CVTColor, DotType, halftone, ResizesAlg
import cv2 as cv


def rgb_halftone(
    img: np.ndarray, dot_size: list[int], angle: list[int], dot_type: list[DotType], scale: float | None, ssaa_filter: ResizesAlg
) -> np.ndarray:
    dot_size_len = len(dot_size)
    angle_len = len(angle)
    dot_type_len = len(dot_type)
    if img.ndim == 2:
        img = cvt_color(img, CVTColor.Gray2RGB)
    img = halftone(
        img,
        [dot_size[index % dot_size_len] for index in range(3)],
        [angle[index % angle_len] for index in range(3)],
        [dot_type[index % dot_type_len] for index in range(3)],
        scale,
        ssaa_filter,
    )
    return img


def cmyk_halftone(
    img: np.ndarray, dot_size: list[int], angle: list[int], dot_type: list[DotType], scale: float | None, ssaa_filter: ResizesAlg
) -> np.ndarray:
    dot_size_len = len(dot_size)
    angle_len = len(angle)
    dot_type_len = len(dot_type)
    if img.ndim == 2:
        img = cvt_color(img, CVTColor.Gray2RGB)
    img = cvt_color(img, CVTColor.RGB2CMYK)
    img = halftone(
        img,
        [dot_size[index % dot_size_len] for index in range(4)],
        [angle[index % angle_len] for index in range(4)],
        [dot_type[index % dot_type_len] for index in range(4)],
        scale,
        ssaa_filter,
    )
    img = cvt_color(img, CVTColor.CMYK2RGB)
    return img


def hsv_halftone(
    img: np.ndarray, dot_size: list[int], angle: list[int], dot_type: list[DotType], scale: float | None, ssaa_filter: ResizesAlg
) -> np.ndarray:
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    img[..., 2] = screentone(img[..., 2], dot_size[0], angle[0], dot_type[0], scale, ssaa_filter)
    img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    return img


def gray_halftone(
    img: np.ndarray, dot_size: list[int], angle: list[int], dot_type: list[DotType], scale: float | None, ssaa_filter: ResizesAlg
) -> np.ndarray:
    if img.ndim == 3:
        img = cvt_color(img, CVTColor.RGB2Gray_2020)
    return screentone(img, dot_size[0], angle[0], dot_type[0], scale, ssaa_filter)


MODE_MAP = {'rgb': rgb_halftone, 'gray': gray_halftone, 'hsv': hsv_halftone, 'cmyk': cmyk_halftone}

Mode = Literal['rgb', 'gray', 'hsv', 'cmyk']
