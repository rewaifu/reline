from __future__ import annotations

import os.path
from dataclasses import dataclass
from typing import List

from pepeline import read, ImgFormat

from reline.static import Node, NodeOptions, ImageFile


@dataclass(frozen=True)
class FileReaderOptions(NodeOptions):
    path: str


class FileReaderNode(Node[FileReaderOptions]):
    def __init__(self, options: FileReaderOptions):
        super().__init__(options)

    def process(self, _) -> List[ImageFile]:
        basename, _ = os.path.splitext(os.path.basename(self.options.path))
        data = read(self.options.path, format=ImgFormat.F32)

        return [ImageFile(data, basename)]
