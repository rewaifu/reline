from dataclasses import dataclass
from typing import Optional, List, Literal

import numpy as np
import torch.cuda
from resselt.utils import ExactTiler, MaxTiler, NoTiling, upscale_with_tiler, empty_cuda_cache
from resselt import global_registry
from pepeline import cvt_color, CvtType
from reline.static import Node, NodeOptions, ImageFile
import logging

Tiler = Literal['exact', 'max', 'no_tiling']


@dataclass(frozen=True)
class UpscaleOptions(NodeOptions):
    model: str
    tiler: Tiler
    exact_tiler_size: Optional[int] = None
    allow_cpu_upscale: Optional[bool] = False


class UpscaleNode(Node[UpscaleOptions]):
    def __init__(self, options: UpscaleOptions):
        super().__init__(options)

        if not torch.cuda.is_available() and not options.allow_cpu_upscale:
            raise 'CUDA is not available. If you want scale with CPU use `allow_cpu_upscale` option'

        state_dict = torch.load(options.model)
        self.model = global_registry.load_from_state_dict(state_dict)
        self.model_parameters = self.model.parameters()
        self.tiler = self._create_tiler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cuda':
            empty_cuda_cache()

    def _img_ch_to_model_ch(self, img: np.ndarray) -> np.ndarray:
        img_shape = img.shape
        img = img.squeeze()
        if self.model_parameters[1] == 3:
            if len(img_shape) == 2:
                img = cvt_color(img, CvtType.GRAY2RGB)
        elif self.model_parameters[1] == 1:
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
                return ExactTiler(self.options.exact_tiler_size)
            case 'max':
                return MaxTiler()
            case 'no_tiling':
                return NoTiling()
            case _:
                raise ValueError(f'Unknown tiler option `{self.options.tiler}`')

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        for file in files:
            img = self._img_ch_to_model_ch(file.data)
            file.data = upscale_with_tiler(img, self.tiler, self.model, self.device).squeeze()
        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        img = self._img_ch_to_model_ch(file.data)
        file.data = upscale_with_tiler(img, self.tiler, self.model, self.device).squeeze()
        return file

    def video_process(self, file: np.ndarray) -> np.ndarray:
        img = self._img_ch_to_model_ch(file)
        # img = fast_color_level(img, 1, 254)
        file = upscale_with_tiler(img, self.tiler, self.model, self.device).squeeze()
        return file
