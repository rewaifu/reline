from __future__ import annotations

import os.path
from dataclasses import dataclass
from enum import Enum
from typing import List

from pepeline import save

from reline.static import Node, NodeOptions, ImageFile


class FileFormat(Enum):
    PNG = 'png'


@dataclass(frozen=True)
class FolderWriterOptions(NodeOptions):
    path: str
    format: FileFormat = FileFormat.PNG


class FolderWriterNode(Node[FolderWriterOptions]):
    def __init__(self, options: FolderWriterOptions):
        super().__init__(options)

    def process(self, files: List[ImageFile]):
        os.makedirs(self.options.path, exist_ok=True)

        for file in files:
            save(file.data, os.path.join(self.options.path, f'{file.basename}.{self.options.format.value}'))
