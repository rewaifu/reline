from typing import Literal, List

import numpy as np

from ...static import NodeOptions, Node, ImageFile
from dataclasses import dataclass
from pepeline import CVTColor, cvt_color

CvtDict = {
    'RGB2Gray2020': CVTColor.RGB2Gray_2020,
    'RGB2Gray709': CVTColor.RGB2Gray_709,
    'RGB2Gray': CVTColor.RGB2Gray_601,
    'Gray2RGB': CVTColor.Gray2RGB,
}
CvtTypeList = Literal['RGB2Gray2020', 'RGB2Gray709', 'RGB2Gray', 'Gray2RGB']


@dataclass(frozen=True)
class CvtColorOptions(NodeOptions):
    cvt_type: CvtTypeList


class CvtColorNode(Node[CvtColorOptions]):
    def __init__(self, options):
        super().__init__(options)
        self.cvt_type = CvtDict[options.cvt_type]

    def __cvt_logic(self, img: np.ndarray) -> np.ndarray:
        if self.cvt_type in [CVTColor.RGB2Gray_2020, CVTColor.RGB2Gray_709, CVTColor.RGB2Gray_601] and img.ndim == 3:
            return cvt_color(img, self.cvt_type)
        elif self.cvt_type in [CVTColor.Gray2RGB] and img.ndim == 2:
            return cvt_color(img, self.cvt_type)
        else:
            return img

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        for file in files:
            file.data = self.__cvt_logic(file.data)

        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        file.data = self.__cvt_logic(file.data)

        return file

    def video_process(self, file: np.ndarray) -> np.ndarray:
        file = self.__cvt_logic(file)

        return file
