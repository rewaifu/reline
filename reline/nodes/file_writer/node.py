from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pepeline import save
from reline.static import Node, NodeOptions, ImageFile


@dataclass(frozen=True)
class FileWriterOptions(NodeOptions):
    path: str


class FileWriterNode(Node[FileWriterOptions]):
    def __init__(self, options: FileWriterOptions):
        super().__init__(options)

    def process(self, files: List[ImageFile]):
        if len(files) != 1:
            raise ValueError('Expected single image file')

        return save(files[0].data, self.options.path)
