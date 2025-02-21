from typing import Literal

import numpy as np
from pepeline import screentone, cvt_color, CvtType, TypeDot
import cv2 as cv


def rgb_halftone(img: np.ndarray, dot_size: list[int], angle: list[int], dot_type: list[TypeDot]) -> np.ndarray:
    dot_size_len = len(dot_size)
    angle_len = len(angle)
    dot_type_len = len(dot_type)
    if img.ndim == 2:
        img = cvt_color(img, CvtType.GRAY2RGB)
    img[..., 0] = screentone(img[..., 0], dot_size[0 % dot_size_len], angle[0 % angle_len], dot_type[0 % dot_type_len])
    img[..., 1] = screentone(img[..., 1], dot_size[1 % dot_size_len], angle[1 % angle_len], dot_type[1 % dot_type_len])
    img[..., 2] = screentone(img[..., 2], dot_size[2 % dot_size_len], angle[2 % angle_len], dot_type[2 % dot_type_len])
    return img


def cmyk_halftone(img: np.ndarray, dot_size: list[int], angle: list[int], dot_type: list[TypeDot]) -> np.ndarray:
    dot_size_len = len(dot_size)
    angle_len = len(angle)
    dot_type_len = len(dot_type)
    if img.ndim == 2:
        img = cvt_color(img, CvtType.GRAY2RGB)
    img = cvt_color(img, CvtType.RGB2CMYK)
    img[..., 0] = screentone(img[..., 0], dot_size[0 % dot_size_len], angle[0 % angle_len], dot_type[0 % dot_type_len])
    img[..., 1] = screentone(img[..., 1], dot_size[1 % dot_size_len], angle[1 % angle_len], dot_type[1 % dot_type_len])
    img[..., 2] = screentone(img[..., 2], dot_size[2 % dot_size_len], angle[2 % angle_len], dot_type[2 % dot_type_len])
    img[..., 3] = screentone(img[..., 3], dot_size[3 % dot_size_len], angle[3 % angle_len], dot_type[3 % dot_type_len])
    img = cvt_color(img, CvtType.CMYK2RGB)
    return img


def hsv_halftone(img: np.ndarray, dot_size: list[int], angle: list[int], dot_type: list[TypeDot]) -> np.ndarray:
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    img[..., 2] = screentone(img[..., 2], dot_size[0], angle[0], dot_type[0])
    img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    return img


def gray_halftone(img: np.ndarray, dot_size: list[int], angle: list[int], dot_type: list[TypeDot]) -> np.ndarray:
    if img.ndim == 3:
        img = cvt_color(img, CvtType.RGB2GrayBt2020)
    img = screentone(img, dot_size[0], angle[0], dot_type[0])
    return img


MODE_MAP = {'rgb': rgb_halftone, 'gray': gray_halftone, 'hsv': hsv_halftone, 'cmyk': cmyk_halftone}

Mode = Literal['rgb', 'gray', 'hsv', 'cmyk']
