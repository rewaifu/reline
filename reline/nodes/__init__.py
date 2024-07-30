from .file_reader import FileReaderNode, FileReaderOptions
from .file_writer import FileWriterNode, FileWriterOptions
from .folder_reader import FolderReaderNode, FolderReaderOptions
from .folder_writer import FolderWriterNode, FolderWriterOptions, FileFormat
from .resize import ResizeNode, ResizeOptions
from .upscale import UpscaleNode, UpscaleOptions
from .level import LevelNode, LevelOptions
from .screentone import ScreentoneNode, ScreentoneOptions
from .sharp import SharpNode, SharpOptions

from .registry import Registry

INTERNAL_REGISTRY = (
    Registry()
    .set('resize', ResizeNode, ResizeOptions)
    .set('file_reader', FileReaderNode, FileReaderOptions)
    .set('file_writer', FileWriterNode, FileWriterOptions)
    .set('upscale', UpscaleNode, UpscaleOptions)
    .set('folder_reader', FolderReaderNode, FolderReaderOptions)
    .set('folder_writer', FolderWriterNode, FolderWriterOptions)
    .set('level', LevelNode, LevelOptions)
    .set('screentone', ScreentoneNode, ScreentoneOptions)
    .set('sharp', SharpNode, SharpOptions)
)

__all__ = [
    'FileReaderNode',
    'FileWriterNode',
    'FileReaderOptions',
    'FileWriterOptions',
    'FolderReaderNode',
    'FolderReaderOptions',
    'FolderWriterNode',
    'FolderWriterOptions',
    'FileFormat',
    'ResizeNode',
    'ResizeOptions',
    'UpscaleNode',
    'UpscaleOptions',
    'LevelNode',
    'LevelOptions',
    'ScreentoneNode',
    'ScreentoneOptions',
    'SharpNode',
    'SharpOptions',
    'INTERNAL_REGISTRY',
]
