from dataclasses import dataclass
from typing import Optional, List, Literal

import numpy as np
import torch.cuda
from resselt import load_from_file
from reutils.tiling import MaxTileSize, ExactTileSize, NoTiling, process_tiles
from pepeline import cvt_color, CvtType
from reline.static import Node, NodeOptions, ImageFile
import logging

Tiler = Literal['exact', 'max', 'no_tiling']
DType = Literal['F32', 'F16', 'BF16']


def empty_cuda_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


@dataclass(frozen=True)
class UpscaleOptions(NodeOptions):
    model: str
    tiler: Tiler
    dtype: Optional[DType] = 'F32'
    exact_tiler_size: Optional[int] = 256
    allow_cpu_upscale: Optional[bool] = False


class UpscaleNode(Node[UpscaleOptions]):
    def __init__(self, options: UpscaleOptions):
        super().__init__(options)

        if not torch.cuda.is_available() and not options.allow_cpu_upscale:
            raise 'CUDA is not available. If you want scale with CPU use `allow_cpu_upscale` option'

        self.model = load_from_file(options.model)
        self.tiler = self._create_tiler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if options.dtype == 'F16':
            self.dtype = torch.half
        elif options.dtype == 'BF16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        if self.device == 'cuda':
            empty_cuda_cache()

    def _img_ch_to_model_ch(self, img: np.ndarray) -> np.ndarray:
        img_shape = img.shape
        img = img.squeeze()
        if self.model.parameters_info.in_channels == 3:
            if len(img_shape) == 2:
                img = cvt_color(img, CvtType.GRAY2RGB)
        elif self.model.parameters_info.in_channels == 1:
            if len(img_shape) == 3:
                img = cvt_color(img, CvtType.RGB2GrayBt2020)
        else:
            logging.error('model format is not currently supported')
        return img

    def _create_tiler(self):
        match self.options.tiler:
            case 'exact':
                if self.options.exact_tiler_size is None:
                    raise ValueError('Exact tiler requires `exact_tiler_size` param')
                return ExactTileSize(self.options.exact_tiler_size)
            case 'max':
                return MaxTileSize()
            case 'no_tiling':
                return NoTiling()
            case _:
                raise ValueError(f'Unknown tiler option `{self.options.tiler}`')

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        for file in files:
            img = self._img_ch_to_model_ch(file.data)
            file.data = process_tiles(
                img, tiler=self.tiler, model=self.model, device=self.device, dtype=self.dtype, scale=self.model.parameters_info.upscale
            )
        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        img = self._img_ch_to_model_ch(file.data)
        file.data = process_tiles(
            img, tiler=self.tiler, model=self.model, device=self.device, dtype=self.dtype, scale=self.model.parameters_info.upscale
        )
        return file

    def video_process(self, file: np.ndarray) -> np.ndarray:
        img = self._img_ch_to_model_ch(file)
        file = process_tiles(img, tiler=self.tiler, model=self.model, device=self.device, dtype=self.dtype, scale=self.model.parameters_info.upscale)
        return file
