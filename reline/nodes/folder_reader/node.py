from __future__ import annotations

import os.path
from dataclasses import dataclass
from typing import List, Optional, Literal
import logging

from pepeline import read, ImgFormat, ImgColor

from reline.static import Node, NodeOptions, ImageFile

MODE_MAP = {'rgb': ImgColor.RGB, 'gray': ImgColor.GRAY, 'dynamic': ImgColor.DYNAMIC}

Mode = Literal['rgb', 'gray', 'dynamic']


@dataclass(frozen=True)
class FolderReaderOptions(NodeOptions):
    path: str
    recursive: Optional[bool] = False
    mode: Optional[Mode] = 'dynamic'


class FolderReaderNode(Node[FolderReaderOptions]):
    def __init__(self, options: FolderReaderOptions):
        super().__init__(options)
        self.mode = MODE_MAP[options.mode]
        self.dir_path = os.path.abspath(self.options.path)

    def _scandir(self, dir_path: str):
        file_paths = []

        try:
            for entry in os.scandir(dir_path):
                if entry.is_file():
                    file_paths.append(os.path.abspath(entry.path))
                elif entry.is_dir() and self.options.recursive:
                    file_paths.extend(self._scandir(os.path.abspath(entry.path)))
        except OSError as e:
            logging.error(f'Error scanning directory {dir_path}: {e}')

        return file_paths

    def process(self, _) -> List[ImageFile]:
        file_paths = self._scandir(self.dir_path)
        files = []
        basename = None
        for file_path in file_paths:
            try:
                commonprefix = os.path.commonprefix([self.dir_path, file_path])
                dirpath = os.path.dirname(os.path.relpath(file_path, commonprefix))
                basename, _ = os.path.splitext(os.path.basename(file_path))

                data = read(file_path, self.mode, ImgFormat.F32)

                file = ImageFile(data, basename, dirpath)
                files.append(file)
            except Exception as e:
                logging.warning(f'image {basename} not decoded due to error: {e}')
                continue

        return files
