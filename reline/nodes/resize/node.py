from dataclasses import dataclass
from typing import List, Optional

from chainner_ext import resize

from reline.static import Node, NodeOptions, ImageFile
from .filter_type import FILTER_MAP, FilterType


@dataclass(frozen=True)
class ResizeOptions(NodeOptions):
    height: Optional[int] = None
    width: Optional[int] = None
    percent: Optional[float] = None
    filter: Optional[FilterType] = 'cubic_catrom'
    gamma_correction: Optional[bool] = False
    spread: Optional[bool] = False
    spread_size: Optional[int] = 2800


class ResizeNode(Node[ResizeOptions]):
    def __init__(self, options: ResizeOptions):
        super().__init__(options)

        if self.options.width is None and self.options.height is None and self.options.percent is None:
            raise ValueError('At least one of width, height, or percent must be specified.')

        self.filter = FILTER_MAP[self.options.filter]
        if options.percent:
            self.calculate_size = self._calculate_size__percent
        else:
            if self.options.width and not self.options.height:
                self.calculate_size = self._calculate_size__width
            elif self.options.height and not self.options.width:
                self.calculate_size = self._calculate_size__height
            else:
                self.calculate_size = self._calculate_size__sides

    def _calculate_size__height(self, height: int, width: int):
        return self.options.height, int(width * (self.options.height / height))

    def _calculate_size__width(self, height: int, width: int):
        if self.options.spread and width > height:
            width = self.options.spread_size
        return int(height * (self.options.width / width)), self.options.width

    def _calculate_size__percent(self, height: int, width: int):
        return int(height * self.options.percent), int(width * self.options.percent)

    def _calculate_size__sides(self, _height: int, _width: int):
        return self.options.height, self.options.width

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        for file in files:
            h, w = self.calculate_size(*file.data.shape[:2])
            file.data = resize(file.data, (w, h), self.filter, self.options.gamma_correction).squeeze()

        return files
