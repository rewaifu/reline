from __future__ import annotations

import os.path
from dataclasses import dataclass, field
from typing import List, Optional

from pepeline import read, ImgFormat

from reline.static import Node, NodeOptions, ImageFile


@dataclass(frozen=True)
class FolderReaderOptions(NodeOptions):
    path: str
    recursive: Optional[bool] = False
    allowed_extensions: Optional[List[str]] = field(default_factory=lambda: ['png', 'jpg', 'jpeg', 'webp'])


class FolderReaderNode(Node[FolderReaderOptions]):
    def __init__(self, options: FolderReaderOptions):
        super().__init__(options)

    def process(self, _) -> List[ImageFile]:
        def scandir(dir_path: str):
            file_paths = []

            try:
                for entry in os.scandir(dir_path):
                    if entry.is_file():
                        for ext in self.options.allowed_extensions:
                            if entry.name.endswith(ext):
                                file_paths.append(os.path.abspath(entry.path))
                    elif entry.is_dir() and self.options.recursive:
                        file_paths.extend(scandir(os.path.abspath(entry.path)))
            except OSError as e:
                print(f'Error scanning directory {dir_path}: {e}')

            return file_paths

        file_paths = scandir(self.options.path)
        files = []

        for file_path in file_paths:
            basename, _ = os.path.splitext(os.path.basename(file_path))
            data = read(file_path, format=ImgFormat.F32)

            file = ImageFile(data, basename)
            files.append(file)

        return files
