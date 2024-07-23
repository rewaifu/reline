from typing import Literal

from chainner_ext.chainner_ext import ResizeFilter

FILTER_MAP = {
    'nearest': ResizeFilter.Nearest,
    'box': ResizeFilter.Box,
    'linear': ResizeFilter.Linear,
    'hermite': ResizeFilter.Hermite,
    'cubic_catrom': ResizeFilter.CubicCatrom,
    'cubic_mitchell': ResizeFilter.CubicMitchell,
    'cubic_bspline': ResizeFilter.CubicBSpline,
    # https://github.com/chaiNNer-org/chaiNNer-rs/issues/28
    #'hamming': ResizeFilter.Hamming,
    #'hann': ResizeFilter.Hann,
    
    'lanczos': ResizeFilter.Lanczos,
    'lagrange': ResizeFilter.Lagrange,
    'gauss': ResizeFilter.Gauss,
}

FilterType = Literal[
    'nearest',
    'box',
    'linear',
    'hermite',
    'cubic_catrom',
    'cubic_mitchell',
    'cubic_bspline',
    #'hamming',
    #'hann',
    'lanczos',
    'lagrange',
    'gauss',
]
