from dataclasses import dataclass
from typing import List, Optional, Literal

from pepeline import TypeDot

from ._halftone_func import MODE_MAP, Mode
from reline.static import Node, NodeOptions, ImageFile

DOT_TYPE_MAP = {
    'circle': TypeDot.CIRCLE,
    'cross': TypeDot.CROSS,
    'ellipse': TypeDot.ELLIPSE,
    'invline': TypeDot.INVLINE,
    'line': TypeDot.LINE,
}

DotType = Literal['line', 'cross', 'ellipse', 'invline', 'line', 'circle']


def _int_to_list(int_value: int | list[int]):
    if isinstance(int_value, int):
        return [int_value]
    return int_value


@dataclass(frozen=True)
class HalftoneOptions(NodeOptions):
    dot_size: Optional[int] | Optional[list[int]] = 7
    angle: Optional[int] | Optional[list[int]] = 0
    dot_type: Optional[DotType] | Optional[list[DotType]] = 'circle'
    halftone_mode: Optional[Mode] = 'gray'


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

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        for file in files:
            file.data = self.halftone(file.data.squeeze(), self.dot_size, self.angle, self.dot_type)
        return files
