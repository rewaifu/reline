from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from pepeline import fast_color_level

from reline.static import Node, NodeOptions, ImageFile


@dataclass(frozen=True)
class LevelOptions(NodeOptions):
    low_input: Optional[int] = 0
    high_input: Optional[int] = 255
    low_output: Optional[int] = 0
    high_output: Optional[int] = 255
    gamma: Optional[float] = 1.0


class LevelNode(Node[LevelOptions]):
    def __init__(self, options):
        super().__init__(options)

    def process(self, files: List[ImageFile]) -> List[ImageFile]:
        for file in files:
            file.data = fast_color_level(
                file.data,
                self.options.low_input,
                self.options.high_input,
                self.options.low_output,
                self.options.high_output,
                self.options.gamma,
            )

        return files

    def single_process(self, file: ImageFile) -> ImageFile:
        file.data = fast_color_level(
            file.data,
            self.options.low_input,
            self.options.high_input,
            self.options.low_output,
            self.options.high_output,
            self.options.gamma,
        )
        return file

    def video_process(self, file: np.ndarray) -> np.ndarray:
        file = fast_color_level(
            file,
            self.options.low_input,
            self.options.high_input,
            self.options.low_output,
            self.options.high_output,
            self.options.gamma,
        )
        return file
