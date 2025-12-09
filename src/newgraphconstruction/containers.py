from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RawGraphStructure:
    """
    Raw graph structure containing edge_index and edge_weight.
    """
    edge_index:  List[Tuple[int, int]]
    edge_weight: List[float]

    def __repr__(self) -> str:
        num_nodes = len({v for edge in self.edge_index for v in edge})
        num_edges = len(self.edge_index)
        return f"<RawGraphStructure(num_nodes={num_nodes}, num_edges={num_edges})>"


@dataclass
class DynamicRawGraphStructure:
    """
    Sequence of RawGraphStructure objects with associated timestamps.
    """
    timestamps:    List[str]
    rawstructures: List[RawGraphStructure]

    def __post_init__(self):
        T = len(self.timestamps)
        if len(self.rawstructures) != T:
            raise ValueError(
                f"Number of rawstructures ({len(self.rawstructures)}) must match timestamps ({T})"
            )

    def __len__(self):
        return len(self.timestamps)

    def __iter__(self):
        return iter(self.rawstructures)

    def __getitem__(self, idx) -> Tuple[RawGraphStructure, str]:
        return self.rawstructures[idx], self.timestamps[idx]

    def __repr__(self):
        return f"<DynamicRawGraphStructure(T={len(self)})>"
