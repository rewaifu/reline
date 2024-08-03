from .file_reader import FileReaderNode, FileReaderOptions
from .file_writer import FileWriterNode, FileWriterOptions
from .folder_reader import FolderReaderNode, FolderReaderOptions
from .folder_writer import FolderWriterNode, FolderWriterOptions, FileFormat
from .resize import ResizeNode, ResizeOptions
from .upscale import UpscaleNode, UpscaleOptions
from .level import LevelNode, LevelOptions
from .halftone import HalftoneNode, HalftoneOptions
from .sharp import SharpNode, SharpOptions
from .dithering import DitheringNode, DitheringOptions

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
    .set('halftone', HalftoneNode, HalftoneOptions)
    .set('sharp', SharpNode, SharpOptions)
    .set('sharp', DitheringNode, DitheringOptions)
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
    'HalftoneNode',
    'HalftoneOptions',
    'SharpNode',
    'SharpOptions',
    'INTERNAL_REGISTRY',
]
