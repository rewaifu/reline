from dataclasses import dataclass
from typing import Optional, List, Literal

import torch.cuda
from resselt.utils import ExactTiler, MaxTiler, NoTiling, upscale_with_tiler
from resselt import global_registry

from reline.static import Node, NodeOptions, ImageFile

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
        self.tiler = self._create_tiler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            file.data = upscale_with_tiler(file.data, self.tiler, self.model, self.device)
        return files
