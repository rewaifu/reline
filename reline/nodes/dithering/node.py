from dataclasses import dataclass
from typing import List, Optional, Literal

import numpy as np

from reline.static import Node, NodeOptions, ImageFile
from chainner_ext import quantize, error_diffusion_dither, ordered_dither, riemersma_dither, DiffusionAlgorithm, UniformQuantization

ERROR_DITHERING_MAP = {
    'floydsteinberg': DiffusionAlgorithm.FloydSteinberg,
    'jarvisjudiceninke': DiffusionAlgorithm.JarvisJudiceNinke,
    'stucki': DiffusionAlgorithm.Stucki,
    'atkinson': DiffusionAlgorithm.Atkinson,
    'burkes': DiffusionAlgorithm.Burkes,
    'sierra': DiffusionAlgorithm.Sierra,
    'tworowsierra': DiffusionAlgorithm.TwoRowSierra,
    'sierraLite': DiffusionAlgorithm.SierraLite,
}
DITHERING_TYPE = Literal[
    'floydsteinberg', 'jarvisjudiceninke', 'stucki', 'atkinson', 'burkes', 'sierra', 'tworowsierra', 'sierraLite', 'order', 'riemersma', 'quantize'
]


@dataclass(frozen=True)
class DitheringOptions(NodeOptions):
    colors_per_ch: Optional[int] = 16
    dith_type: Optional[DITHERING_TYPE] = 'sierra'
    map_size: Optional[int] = 8
    history: Optional[int] = 10
    ratio: Optional[float] = 0.5


class DitheringNode(Node[DitheringOptions]):
    def __init__(self, options):
        super().__init__(options)
        self.UQ = UniformQuantization(options.colors_per_ch)

    def _error(self, img: np.ndarray) -> np.ndarray:
        return error_diffusion_dither(img, self.UQ, ERROR_DITHERING_MAP[self.options.dith_type]).squeeze()

    def _order(self, img: np.ndarray) -> np.ndarray:
        return ordered_dither(img, self.UQ, self.options.map_size).squeeze()

    def _riemersma(self, img: np.ndarray) -> np.ndarray:
        return riemersma_dither(img, self.UQ, self.options.history, self.options.ratio).squeeze()

    def _quantize(self, img: np.ndarray) -> np.ndarray:
        return quantize(img, self.UQ).squeeze()

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        dithering_type_map = {
            'floydsteinberg': lambda img: self._error(img),
            'jarvisjudiceninke': lambda img: self._error(img),
            'stucki': lambda img: self._error(img),
            'atkinson': lambda img: self._error(img),
            'burkes': lambda img: self._error(img),
            'sierra': lambda img: self._error(img),
            'tworowsierra': lambda img: self._error(img),
            'sierraLite': lambda img: self._error(img),
            'order': lambda img: self._order(img),
            'riemersma': lambda img: self._riemersma(img),
            'quantize': lambda img: self._quantize(img),
        }
        dithering_process = dithering_type_map[self.options.dith_type]
        for file in files:
            file.data = dithering_process(img=file.data)

        return files
