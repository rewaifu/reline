from __future__ import annotations
from typing import List, Dict
from tqdm import tqdm

from ..nodes import INTERNAL_REGISTRY
from ..static import Node


class Pipeline:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def process(self, with_tqdm: bool = True):
        data = []
        for node in tqdm(self.nodes, desc='Node Processing', disable=not with_tqdm):
            data = node.process(data)
        return data

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
