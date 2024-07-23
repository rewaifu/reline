from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import torch.cuda
from resselt.utils import AutoTiler, ExactTiler, MaxTiler, NoTiling, upscale_with_tiler
from resselt import global_registry

from reline.static import Node, NodeOptions, ImageFile


class Tiler(Enum):
    Auto = 'auto'
    Exact = 'exact'
    Max = 'max'
    No = 'no_tiling'


@dataclass(frozen=True)
class UpscaleOptions(NodeOptions):
    weights_path: str
    tiler: Tiler
    exact_tiler_size: Optional[int] = None
    allow_cpu_upscale: Optional[bool] = None


class UpscaleNode(Node[UpscaleOptions]):
    def __init__(self, options: UpscaleOptions):
        super().__init__(options)

        state_dict = torch.load(options.weights_path)
        self.model = global_registry.load_from_state_dict(state_dict)
        self.tiler = self._create_tiler()

    def _create_tiler(self):
        match self.options.tiler:
            case Tiler.Auto:
                return AutoTiler(self.model)
            case Tiler.Exact:
                if self.options.exact_tiler_size is None:
                    raise ValueError('Exact tiler requires `exact_tiler_size` param')
                return ExactTiler(self.options.exact_tiler_size)
            case Tiler.Max:
                return MaxTiler()
            case Tiler.No:
                return NoTiling()
            case _:
                raise ValueError(f'Unknown tiler option `{self.options.tiler}`')

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        if not torch.cuda.is_available() and not self.options.allow_cpu_upscale:
            raise 'CUDA is not available. If you want scale with CPU use `allow_cpu_upscale` option'

        for file in files:
            file.data = upscale_with_tiler(file.data, self.tiler, self.model, device=torch.device('cuda'))
        return files
