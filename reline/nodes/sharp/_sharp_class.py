import cv2 as cv
import numpy as np

from pepeline import color_levels


class Canny:
    def __init__(self, canny_type):
        self.canny_type = canny_type

    def run(self, img_float: np.ndarray) -> np.ndarray:
        image = (img_float * 255).astype(np.uint8)
        edges = np.clip(255 - cv.Canny(image, 750, 800, apertureSize=3, L2gradient=False), 0, 1)
        if self.canny_type == 'unsharp':
            kernel = np.ones((5, 5), dtype=np.uint8)
            edges = cv.dilate(edges, kernel, iterations=1)
            blurred = cv.GaussianBlur(img_float, (0, 0), sigmaX=0.5, sigmaY=0.5, borderType=cv.BORDER_REFLECT)
            white = ((img_float * edges) > 0.9882352941176471).astype(np.float32)
            black = (img_float < 0.011764705882352941).astype(np.float32) * edges
            black = cv.dilate(black, kernel, iterations=1)
            white = cv.dilate(white, kernel, iterations=1)
            w_b = ((white + black) - 1).clip(0, 1)
            img_float = (cv.addWeighted(img_float, 11, blurred, -10, 0) * w_b + img_float * (1 - w_b)).clip(0, 1)

            return img_float
        else:
            return np.where(edges, img_float, 1.0 if self.canny_type == 'invert' else 0.0)


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
        black_mask = (img_float > self.diapason).astype(np.float32)
        blur = cv.GaussianBlur(black_mask, (3, 3), 0)
        blur_mask = (blur > 0.9).astype(np.float32)
        return np.where(blur_mask, img_float, 0)


class ColorLevels:
    def __init__(self, low_input, high_input, gamma):
        self.low_input = low_input
        self.high_input = high_input
        self.gamma = gamma

    def run(self, img_float: np.ndarray) -> np.ndarray:
        return color_levels(img_float, self.low_input, self.high_input, 0, 255, self.gamma)
