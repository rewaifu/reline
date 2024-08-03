from __future__ import annotations

import os.path
from dataclasses import dataclass
from typing import List, Optional, Literal

from pepeline import read, ImgFormat, ImgColor

from reline.static import Node, NodeOptions, ImageFile

MODE_MAP = {'rgb': ImgColor.RGB, 'gray': ImgColor.GRAY, 'dynamic': ImgColor.DYNAMIC}

Mode = Literal['rgb', 'gray', 'dynamic']


@dataclass(frozen=True)
class FileReaderOptions(NodeOptions):
    path: str
    mode: Optional[Mode] = 'dynamic'


class FileReaderNode(Node[FileReaderOptions]):
    def __init__(self, options: FileReaderOptions):
        super().__init__(options)
        self.mode = MODE_MAP[options.mode]

    def process(self, _) -> List[ImageFile]:
        basename, _ = os.path.splitext(os.path.basename(self.options.path))
        data = read(self.options.path, mode=self.mode, format=ImgFormat.F32)

        return [ImageFile(data, basename)]
