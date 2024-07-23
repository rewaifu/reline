from dataclasses import dataclass
from typing import List, Optional, Literal
from pepeline import screentone, TypeDot

from reline.static import Node, NodeOptions, ImageFile

DOT_TYPE_MAP = {
    'circle': TypeDot.CIRCLE,
    'cross': TypeDot.CROSS,
    'ellipse': TypeDot.ELLIPSE,
    'invline': TypeDot.INVLINE,
    'line': TypeDot.LINE,
}

DotType = Literal['line', 'cross', 'ellipse', 'invline', 'line']


@dataclass(frozen=True)
class ScreentoneOptions(NodeOptions):
    dot_size: Optional[int] = 7
    angle: Optional[int] = 0
    dot_type: Optional[DotType] = 'circle'


class ScreentoneNode(Node[ScreentoneOptions]):
    def __init__(self, options):
        super().__init__(options)
        self.dot_type = DOT_TYPE_MAP[options.dot_type]

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        for file in files:
            file.data = screentone(file.data, self.options.dot_size, self.options.angle, self.dot_type)
        return files
