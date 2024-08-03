from __future__ import annotations
from typing import List, Dict
from tqdm import tqdm

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
            nodes_index += 1
            if isinstance(node, FileReaderNode | FolderReaderNode):
                data = node.process(data)
                for img in tqdm(data, desc="Processing Images", disable=not with_tqdm):
                    img = [img]
                    local_node_index = nodes_index
                    for node in self.nodes[local_node_index:]:
                        img = node.process(img)
                        local_node_index += 1
                        if isinstance(node, FolderWriterNode | FileWriterNode):
                            save_index = local_node_index
                            break

                nodes_index = save_index

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
