from __future__ import annotations

import os
from typing import List, Dict

import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip

from ..nodes import INTERNAL_REGISTRY
from ..static import Node
from ..nodes.file_reader import FileReaderNode
from ..nodes.folder_reader import FolderReaderNode
from ..nodes.file_writer import FileWriterNode
from ..nodes.folder_writer import FolderWriterNode


class Pipeline:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def process(self, with_tqdm: bool = True):
        data = []
        for node in tqdm(self.nodes, desc='Node Processing', disable=not with_tqdm):
            data = node.process(data)
        return data

    def process_linear(self, with_tqdm: bool = True):
        data = []
        nodes_index = 0
        save_index = 0
        while nodes_index < len(self.nodes):
            node = self.nodes[nodes_index]
            if isinstance(node, FileReaderNode | FolderReaderNode):
                data = node.single_process(data)
                for img in tqdm(data, desc='Processing Images', disable=not with_tqdm):
                    if img is None:
                        continue
                    local_node_index = nodes_index + 1
                    for node in self.nodes[local_node_index:]:
                        img = node.single_process(img)
                        local_node_index += 1
                        if isinstance(node, FolderWriterNode | FileWriterNode):
                            save_index = local_node_index - 1
                            break
                nodes_index = save_index
                nodes_index += 1
            else:
                nodes_index += 1
        del data

    def process_frame(self, frame):
        frame = frame.astype(np.float32) / 255
        for node in self.nodes:
            frame = node.video_process(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return np.clip(frame * 255, 0, 255).astype(np.uint8)

    def process_video(self, in_folder, out_folder, codec):
        video = VideoFileClip(in_folder).to_RGB()
        processed_video = video.fl_image(self.process_frame)
        audio = video.audio
        final_video = processed_video.set_audio(audio)
        os.makedirs(os.path.dirname(out_folder), exist_ok=True)
        final_video.write_videofile(out_folder, codec=codec)

    @classmethod
    def from_json(cls, data: Dict) -> Pipeline:
        nodes = []

        for item in data:
            node_type = item['type']
            options_data = item['options']

            node_pair = INTERNAL_REGISTRY.get(node_type)
            options = node_pair.options(**options_data)

            node = node_pair.node(options)
            nodes.append(node)

        return Pipeline(nodes)
