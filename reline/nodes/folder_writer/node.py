from __future__ import annotations

import os.path
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from pepeline import save

from reline.static import Node, NodeOptions, ImageFile

FileFormat = Literal['png', 'jpeg']


@dataclass(frozen=True)
class FolderWriterOptions(NodeOptions):
    path: str
    format: FileFormat = 'png'


class FolderWriterNode(Node[FolderWriterOptions]):
    def __init__(self, options: FolderWriterOptions):
        super().__init__(options)

        os.makedirs(options.path, exist_ok=True)

    def process(self, files: List[ImageFile]):
        for file in files:
            full_path = os.path.join(os.path.abspath(self.options.path), file.dir, f'{file.basename}.{self.options.format}')
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            save(file.data, full_path)

    def single_process(self, file: ImageFile):
        full_path = os.path.join(os.path.abspath(self.options.path), file.dir, f'{file.basename}.{self.options.format}')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        save(file.data, full_path)

    def video_process(self, file: np.ndarray) -> np.ndarray:
        raise ValueError('Video scale does not support folder writer')
