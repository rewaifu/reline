from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from pepeline import cvt_color, CvtType
from ._sharp_class import Canny, DiapasonBlack, DiapasonWhite, ColorLevels
from reline.static import Node, NodeOptions, ImageFile


@dataclass(frozen=True)
class SharpOptions(NodeOptions):
    low_input: Optional[int] = 0
    high_input: Optional[int] = 255
    gamma: Optional[float] = 1.0
    diapason_white: Optional[int] = -1
    diapason_black: Optional[int] = -1
    canny: Optional[bool] = False


class SharpNode(Node[SharpOptions]):
    def __init__(self, options):
        super().__init__(options)
        self.stack = []
        if options.low_input != 0 or options.high_input != 255 or options.gamma != 1.0:
            self.stack.append(ColorLevels(options.low_input, options.high_input, options.gamma))
        if options.diapason_white >= 0:
            self.stack.append(DiapasonWhite(options.diapason_white))
        if options.diapason_black >= 0:
            self.stack.append(DiapasonBlack(options.diapason_black))
        if options.canny:
            self.stack.append(Canny())

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        if len(self.stack) == 0:
            return files
        for file in files:
            img_float = file.data.squeeze()
            if img_float.ndim == 3:
                img_float = cvt_color(img_float, CvtType.RGB2GrayBt2020)
            for process in self.stack:
                img_float = process.run(img_float)
            file.data = img_float
        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        if len(self.stack) == 0:
            return file

        img_float = file.data.squeeze()
        if img_float.ndim == 3:
            img_float = cvt_color(img_float, CvtType.RGB2GrayBt2020)
        for process in self.stack:
            img_float = process.run(img_float)
        file.data = img_float
        return file

    def video_process(self, file: np.ndarray) -> np.ndarray:
        if len(self.stack) == 0:
            return file

        img_float = file.squeeze()
        if img_float.ndim == 3:
            img_float = cvt_color(img_float, CvtType.RGB2GrayBt2020)
        for process in self.stack:
            img_float = process.run(img_float)
        return img_float
