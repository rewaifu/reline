from dataclasses import dataclass
from typing import List, Optional, Literal

import numpy as np
from pepeline import DotType

from ._halftone_func import MODE_MAP, Mode
from reline.static import Node, NodeOptions, ImageFile
from ..resize.filter_type import FilterType, FILTER_MAP

DOT_TYPE_MAP = {
    'circle': DotType.CIRCLE,
    'cross': DotType.CROSS,
    'ellipse': DotType.ELLIPSE,
    'invline': DotType.INVLINE,
    'line': DotType.LINE,
}

TypeDot = Literal['line', 'cross', 'ellipse', 'invline', 'circle']


def _int_to_list(int_value: int | list[int]):
    if isinstance(int_value, int):
        return [int_value]
    return int_value


@dataclass(frozen=True)
class HalftoneOptions(NodeOptions):
    dot_size: Optional[int] | Optional[list[int]] = 7
    angle: Optional[int] | Optional[list[int]] = 0
    dot_type: Optional[TypeDot] | Optional[list[TypeDot]] = 'circle'
    halftone_mode: Optional[Mode] = 'gray'
    ssaa_scale: Optional[float] = None
    ssaa_filter: Optional[FilterType] = 'shamming4'


class HalftoneNode(Node[HalftoneOptions]):
    def __init__(self, options):
        super().__init__(options)
        self.dot_size = _int_to_list(options.dot_size)
        self.angle = _int_to_list(options.angle)
        self.halftone = MODE_MAP[options.halftone_mode]
        if isinstance(options.dot_type, str):
            self.dot_type = [DOT_TYPE_MAP[options.dot_type]]
        else:
            self.dot_type = [DOT_TYPE_MAP[dot_type] for dot_type in options.dot_type]
        self.scale = options.ssaa_scale
        self.ssaa_filter = FILTER_MAP[options.ssaa_filter]

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        for file in files:
            file.data = self.halftone(file.data.squeeze(), self.dot_size, self.angle, self.dot_type, self.scale, self.ssaa_filter)
        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        file.data = self.halftone(file.data.squeeze(), self.dot_size, self.angle, self.dot_type, self.scale, self.ssaa_filter)
        return file

    def video_process(self, file: np.ndarray) -> np.ndarray:
        file = self.halftone(file.squeeze(), self.dot_size, self.angle, self.dot_type, self.scale, self.ssaa_filter)
        return file
