import cv2 as cv
import numpy as np
from chainner_ext import binary_threshold
from pepeline import fast_color_level


class Canny:
    @staticmethod
    def run(img_float: np.ndarray) -> np.ndarray:
        image = (img_float * 255).astype(np.uint8)
        edges = np.clip(255 - cv.Canny(image, 750, 800, apertureSize=3, L2gradient=False), 0, 1)
        return np.where(edges, img_float, 0)


class DiapasonWhite:
    def __init__(self, diapason_white: int):
        self.diapason = diapason_white / 255

    def run(self, img_float: np.ndarray) -> np.ndarray:
        image = (img_float * 255).astype(np.uint8)
        median_image = cv.medianBlur(image, 3)
        mask = median_image <= 255 - self.diapason
        return np.where(mask, img_float, 1.0)


class DiapasonBlack:
    def __init__(self, diapason_black: int):
        self.diapason = diapason_black / 255

    def run(self, img_float: np.ndarray) -> np.ndarray:
        black_mask = binary_threshold(img_float, self.diapason, False)
        blur = cv.GaussianBlur(black_mask, (3, 3), 0)
        blur_mask = 1 - binary_threshold(blur, 0.6, False)

        return np.clip(img_float - blur_mask, 0, 1)


class ColorLevels:
    def __init__(self, low_input, high_input, gamma):
        self.low_input = low_input
        self.high_input = high_input
        self.gamma = gamma

    def run(self, img_float: np.ndarray) -> np.ndarray:
        return fast_color_level(img_float, self.low_input, self.high_input, 0, 255, self.gamma)
