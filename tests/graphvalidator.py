from src.dataloading.deepdataloader import GNNDataLoader
import torch
from typing import Dict
import os 
import json

class GraphValidator:
    """
    Comprehensive validation framework for testing graph structure utilization
    """
    
    def __init__(self, 
                 dataloader: GNNDataLoader,
                 graphname: str):
        self.dataloader = dataloader
        self.logs    = {}
        self.graphname = graphname


    def _check_connectivity(self, edge_index: torch.Tensor, num_nodes: int) -> bool:
        """
        Check if graph is connected using BFS
        """
        if num_nodes == 0:
            return False
            
        # Build adjacency list
        adj_list = {i: [] for i in range(num_nodes)}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)
            adj_list[dst].append(src)  # Undirected
        
        # BFS from node 0
        visited = set()
        queue = [0]
        visited.add(0)
        
        while queue:
            node = queue.pop(0)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == num_nodes

    def validate_graph_structure(self):
        """
        Validate basic graph structure properties
        """
        edge_index = self.dataloader.edge_index
        edge_weight= self.dataloader.edge_weight

        num_nodes = torch.unique(edge_index).shape[0]
        num_edges = edge_index.shape[1]
        
        # Check for identity graph
        self_loops = (edge_index[0] == edge_index[1]).sum().item()
        is_identity = self_loops == num_edges
        
        # Calculate density
        max_possible = num_nodes * (num_nodes - 1)
        density = num_edges / max_possible if max_possible > 0 else 0
        
        # Check connectivity
        is_connected = self._check_connectivity(edge_index, num_nodes)
        
        # Edge weight analysis
        weight_stats = {}
        if edge_weight is not None:
            weight_stats = {
                'min': edge_weight.min().item(),
                'max': edge_weight.max().item(),
                'mean': edge_weight.mean().item(),
                'std': edge_weight.std().item(),
                'non_zero': (edge_weight > 0).sum().item()
            }
        
        validation_results = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'self_loops': self_loops,
            'is_identity': is_identity,
            'is_connected': is_connected,
            'weight_stats': weight_stats
        }
        
        # Print warnings
        # if is_identity:
        #     print("⚠️  WARNING: Identity graph detected - no spatial information!")
        # if density > 0.5:
        #     print(f"⚠️  WARNING: Very dense graph (density={density:.3f}) - may cause over-smoothing")
        # elif density < 0.01:
        #     print(f"⚠️  WARNING: Very sparse graph (density={density:.3f}) - may have disconnected components")
        # if not is_connected:
        #     print("⚠️  WARNING: Graph is not connected - some nodes may be isolated")
            
        self.logs = validation_results    


    def run_tests(self,
                  logs_dir = 'tests/logs'):

        self.validate_graph_structure()

        file = os.path.join(logs_dir, f'graph_validation_{self.graphname}.json')

        with open(file, 'w') as f:
                json.dump(tensor_to_python(self.logs), f, indent=2)     

        print('logs saved')   



def tensor_to_python(obj):
    if isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(i) for i in obj]
    elif hasattr(obj, 'item'):  # tensor or scalar with item()
        return obj.item()
    else:
        return obj        