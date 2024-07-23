from __future__ import annotations
from typing import NamedTuple, TypeVar, Dict, Iterator, Type

from ..static import Node, NodeOptions

N = TypeVar('N', bound=Node, covariant=True)
P = TypeVar('P', bound=NodeOptions, covariant=True)


class NodePair(NamedTuple):
    node: Type[N]
    options: Type[P]


class NodePairNotFound(Exception):
    pass


class Registry:
    def __init__(self):
        self.store: Dict[str, NodePair] = {}

    def __contains__(self, name: str):
        return name in self.store

    def __iter__(self) -> Iterator[NodePair]:
        self._iter_keys = iter(self.store)
        return self

    def __next__(self) -> NodePair:
        if self._iter_keys is None:
            raise StopIteration
        try:
            key = next(self._iter_keys)
            return self.store[key]
        except StopIteration:
            self._iter_keys = None
            raise

    def set(self, name: str, node: Type[Node], options: Type[P]) -> Registry:
        self.store[name] = NodePair(node, options)
        return self

    def get(self, name: str) -> NodePair:
        node = self.store[name]
        if not node:
            raise NodePairNotFound
        return node
