from dataclasses import dataclass
from typing import List

from chainner_ext import resize, ResizeFilter

from reline.static import Node, NodeOptions, ImageFile


@dataclass(frozen=True)
class ResizeOptions(NodeOptions):
    height: int
    width: int


class ResizeNode(Node[ResizeOptions]):
    def __init__(self, options: ResizeOptions):
        super().__init__(options)

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        for file in files:
            file.data = resize(file.data, (self.options.width, self.options.height), ResizeFilter.CubicCatrom, False)
        return files
