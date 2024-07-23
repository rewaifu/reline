from __future__ import annotations

import os.path
from dataclasses import dataclass, field
from typing import List, Optional, Literal

from pepeline import read, ImgFormat, ImgColor

from reline.static import Node, NodeOptions, ImageFile

MODE_MAP = {'rgb': ImgColor.RGB, 'gray': ImgColor.GRAY, 'dynamic': ImgColor.DYNAMIC}

Mode = Literal['rgb', 'gray', 'dynamic']


@dataclass(frozen=True)
class FolderReaderOptions(NodeOptions):
    path: str
    recursive: Optional[bool] = False
    allowed_extensions: Optional[List[str]] = field(default_factory=lambda: ['png', 'jpg', 'jpeg', 'webp'])
    mode: Optional[Mode] = 'dynamic'


class FolderReaderNode(Node[FolderReaderOptions]):
    def __init__(self, options: FolderReaderOptions):
        super().__init__(options)
        self.mode = MODE_MAP[options.mode]

    def _scandir(self, dir_path: str):
        file_paths = []

        try:
            for entry in os.scandir(dir_path):
                if entry.is_file():
                    for ext in self.options.allowed_extensions:
                        if entry.name.endswith(ext):
                            file_paths.append(os.path.abspath(entry.path))
                elif entry.is_dir() and self.options.recursive:
                    file_paths.extend(self._scandir(os.path.abspath(entry.path)))
        except OSError as e:
            print(f'Error scanning directory {dir_path}: {e}')

        return file_paths

    def process(self, _) -> List[ImageFile]:
        file_paths = self._scandir(self.options.path)
        files = []

        for file_path in file_paths:
            basename, _ = os.path.splitext(os.path.basename(file_path))
            data = read(file_path, self.mode, ImgFormat.F32)

            file = ImageFile(data, basename)
            files.append(file)

        return files
