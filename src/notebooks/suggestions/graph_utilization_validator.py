"""
Graph Utilization Validation Framework

This module provides comprehensive tests to validate that GNN models
are actually using the graph structure and not just learning node-independent patterns.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloading.gnndataloader import GNNDataLoader
from models.temporal_gcn import TemporalGCNModel
from models.spatial_gcn import SpatialGCNModel
from models.gnnlstmgat import GATLSTMModel
from models.node_lstm import NodeLSTM


class GraphUtilizationValidator:
    """
    Comprehensive validation framework for testing graph structure utilization
    """
    
    def __init__(self, dataloader: GNNDataLoader):
        self.dataloader = dataloader
        self.results = {}
        
    def validate_graph_structure(self, edge_index: torch.Tensor, 
                                edge_weight: Optional[torch.Tensor] = None) -> Dict:
        """
        Validate basic graph structure properties
        """
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
        if is_identity:
            print("⚠️  WARNING: Identity graph detected - no spatial information!")
        if density > 0.5:
            print(f"⚠️  WARNING: Very dense graph (density={density:.3f}) - may cause over-smoothing")
        elif density < 0.01:
            print(f"⚠️  WARNING: Very sparse graph (density={density:.3f}) - may have disconnected components")
        if not is_connected:
            print("⚠️  WARNING: Graph is not connected - some nodes may be isolated")
            
        return validation_results
    
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
    
    def test_identity_graph_comparison(self, model_class, model_name: str, 
                                     test_sample_idx: int = 0) -> Dict:
        """
        Test 1: Compare predictions with identity graph vs. actual graph
        This is the most important test - models should behave differently
        """
        print(f"\n🔍 Testing {model_name} - Identity Graph Comparison")
        
        # Get test sample
        test_sample = self.dataloader.dataset_test[test_sample_idx]
        num_nodes = test_sample.x.shape[0]
        
        # Create identity graph (self-loops only)
        identity_edge_index = torch.stack([
            torch.arange(num_nodes),
            torch.arange(num_nodes)
        ])
        
        # Initialize model
        model = model_class(self.dataloader, name=f"{model_name}_test")
        model.set_model_hparams()
        model.model.eval()
        
        with torch.no_grad():
            # Prediction with actual graph
            pred_actual = model.model(test_sample.x, test_sample.edge_index, test_sample.edge_weight)
            
            # Prediction with identity graph
            pred_identity = model.model(test_sample.x, identity_edge_index, None)
        
        # Calculate similarity metrics
        cosine_sim = torch.cosine_similarity(pred_actual, pred_identity, dim=0)
        mse_diff = torch.mean((pred_actual - pred_identity)**2)
        correlation = torch.corrcoef(torch.stack([pred_actual, pred_identity]))[0, 1]
        
        # Determine if model is using graph structure
        is_using_graph = cosine_sim < 0.9 and mse_diff > 0.01
        
        results = {
            'model_name': model_name,
            'cosine_similarity': cosine_sim.item(),
            'mse_difference': mse_diff.item(),
            'correlation': correlation.item(),
            'is_using_graph': is_using_graph,
            'pred_actual_mean': pred_actual.mean().item(),
            'pred_identity_mean': pred_identity.mean().item(),
            'pred_actual_std': pred_actual.std().item(),
            'pred_identity_std': pred_identity.std().item()
        }
        
        print(f"  Cosine Similarity: {cosine_sim:.4f}")
        print(f"  MSE Difference: {mse_diff:.4f}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Using Graph Structure: {'✅ YES' if is_using_graph else '❌ NO'}")
        
        return results
    
    def test_graph_permutation_invariance(self, model_class, model_name: str,
                                        test_sample_idx: int = 0) -> Dict:
        """
        Test 2: Check if model is permutation invariant
        Predictions should change when node order changes (if using graph structure)
        """
        print(f"\n🔍 Testing {model_name} - Graph Permutation Invariance")
        
        test_sample = self.dataloader.dataset_test[test_sample_idx]
        
        # Create random permutation
        num_nodes = test_sample.x.shape[0]
        perm = torch.randperm(num_nodes)
        
        # Permute features and edge indices
        x_perm = test_sample.x[perm]
        edge_index_perm = torch.zeros_like(test_sample.edge_index)
        for i in range(test_sample.edge_index.shape[1]):
            edge_index_perm[0, i] = perm[test_sample.edge_index[0, i]]
            edge_index_perm[1, i] = perm[test_sample.edge_index[1, i]]
        
        # Initialize model
        model = model_class(self.dataloader, name=f"{model_name}_perm_test")
        model.set_model_hparams()
        model.model.eval()
        
        with torch.no_grad():
            # Original prediction
            pred_original = model.model(test_sample.x, test_sample.edge_index, test_sample.edge_weight)
            
            # Permuted prediction
            pred_permuted = model.model(x_perm, edge_index_perm, test_sample.edge_weight)
        
        # Inverse permute the permuted prediction
        pred_permuted_inv = pred_permuted[torch.argsort(perm)]
        
        # Calculate similarity
        cosine_sim = torch.cosine_similarity(pred_original, pred_permuted_inv, dim=0)
        mse_diff = torch.mean((pred_original - pred_permuted_inv)**2)
        
        # For graph-aware models, predictions should be similar after inverse permutation
        is_permutation_invariant = cosine_sim > 0.95 and mse_diff < 0.01
        
        results = {
            'model_name': model_name,
            'cosine_similarity': cosine_sim.item(),
            'mse_difference': mse_diff.item(),
            'is_permutation_invariant': is_permutation_invariant
        }
        
        print(f"  Cosine Similarity: {cosine_sim:.4f}")
        print(f"  MSE Difference: {mse_diff:.4f}")
        print(f"  Permutation Invariant: {'✅ YES' if is_permutation_invariant else '❌ NO'}")
        
        return results
    
    def test_spatial_diversity(self, model_class, model_name: str,
                             test_sample_idx: int = 0) -> Dict:
        """
        Test 3: Check if model produces spatially diverse predictions
        """
        print(f"\n🔍 Testing {model_name} - Spatial Diversity")
        
        test_sample = self.dataloader.dataset_test[test_sample_idx]
        
        # Initialize model
        model = model_class(self.dataloader, name=f"{model_name}_diversity_test")
        model.set_model_hparams()
        model.model.eval()
        
        with torch.no_grad():
            predictions = model.model(test_sample.x, test_sample.edge_index, test_sample.edge_weight)
        
        # Calculate diversity metrics
        pred_variance = torch.var(predictions).item()
        pred_std = torch.std(predictions).item()
        pred_range = (predictions.max() - predictions.min()).item()
        
        # Check if predictions are too similar (indicating over-smoothing)
        is_diverse = pred_variance > 0.01 and pred_std > 0.1
        
        results = {
            'model_name': model_name,
            'prediction_variance': pred_variance,
            'prediction_std': pred_std,
            'prediction_range': pred_range,
            'is_spatially_diverse': is_diverse
        }
        
        print(f"  Prediction Variance: {pred_variance:.4f}")
        print(f"  Prediction Std: {pred_std:.4f}")
        print(f"  Prediction Range: {pred_range:.4f}")
        print(f"  Spatially Diverse: {'✅ YES' if is_diverse else '❌ NO'}")
        
        return results
    
    def test_edge_weight_sensitivity(self, model_class, model_name: str,
                                   test_sample_idx: int = 0) -> Dict:
        """
        Test 4: Check if model is sensitive to edge weights
        """
        print(f"\n🔍 Testing {model_name} - Edge Weight Sensitivity")
        
        test_sample = self.dataloader.dataset_test[test_sample_idx]
        
        # Create modified edge weights
        if test_sample.edge_weight is not None:
            # Scale weights by 2x
            weights_scaled = test_sample.edge_weight * 2.0
            
            # Randomize weights
            weights_random = torch.rand_like(test_sample.edge_weight)
            
            # Initialize model
            model = model_class(self.dataloader, name=f"{model_name}_weight_test")
            model.set_model_hparams()
            model.model.eval()
            
            with torch.no_grad():
                pred_original = model.model(test_sample.x, test_sample.edge_index, test_sample.edge_weight)
                pred_scaled = model.model(test_sample.x, test_sample.edge_index, weights_scaled)
                pred_random = model.model(test_sample.x, test_sample.edge_index, weights_random)
            
            # Calculate sensitivity
            sensitivity_scaled = torch.mean((pred_original - pred_scaled)**2).item()
            sensitivity_random = torch.mean((pred_original - pred_random)**2).item()
            
            is_weight_sensitive = sensitivity_scaled > 0.001 or sensitivity_random > 0.01
            
            results = {
                'model_name': model_name,
                'sensitivity_scaled': sensitivity_scaled,
                'sensitivity_random': sensitivity_random,
                'is_weight_sensitive': is_weight_sensitive
            }
            
            print(f"  Sensitivity (2x weights): {sensitivity_scaled:.4f}")
            print(f"  Sensitivity (random weights): {sensitivity_random:.4f}")
            print(f"  Weight Sensitive: {'✅ YES' if is_weight_sensitive else '❌ NO'}")
            
        else:
            results = {
                'model_name': model_name,
                'sensitivity_scaled': 0.0,
                'sensitivity_random': 0.0,
                'is_weight_sensitive': False
            }
            print("  No edge weights available for testing")
        
        return results
    
    def run_comprehensive_validation(self, models_to_test: List[Tuple], 
                                   test_sample_idx: int = 0) -> Dict:
        """
        Run all validation tests on specified models
        """
        print("🚀 Starting Comprehensive Graph Utilization Validation")
        print("=" * 60)
        
        # First, validate graph structure
        print("\n📊 Graph Structure Validation")
        graph_validation = self.validate_graph_structure(
            self.dataloader.edge_index, 
            self.dataloader.edge_weight
        )
        
        all_results = {
            'graph_validation': graph_validation,
            'model_tests': {}
        }
        
        # Run tests for each model
        for model_class, model_name in models_to_test:
            print(f"\n{'='*20} Testing {model_name} {'='*20}")
            
            model_results = {}
            
            try:
                # Test 1: Identity graph comparison
                model_results['identity_comparison'] = self.test_identity_graph_comparison(
                    model_class, model_name, test_sample_idx
                )
                
                # Test 2: Permutation invariance
                model_results['permutation_invariance'] = self.test_graph_permutation_invariance(
                    model_class, model_name, test_sample_idx
                )
                
                # Test 3: Spatial diversity
                model_results['spatial_diversity'] = self.test_spatial_diversity(
                    model_class, model_name, test_sample_idx
                )
                
                # Test 4: Edge weight sensitivity
                model_results['edge_weight_sensitivity'] = self.test_edge_weight_sensitivity(
                    model_class, model_name, test_sample_idx
                )
                
            except Exception as e:
                print(f"❌ Error testing {model_name}: {str(e)}")
                model_results['error'] = str(e)
            
            all_results['model_tests'][model_name] = model_results
        
        # Summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict):
        """
        Print a summary of all validation results
        """
        print("\n" + "="*60)
        print("📋 VALIDATION SUMMARY")
        print("="*60)
        
        graph_val = results['graph_validation']
        print(f"\nGraph Structure:")
        print(f"  Nodes: {graph_val['num_nodes']}")
        print(f"  Edges: {graph_val['num_edges']}")
        print(f"  Density: {graph_val['density']:.4f}")
        print(f"  Is Identity: {'❌ YES' if graph_val['is_identity'] else '✅ NO'}")
        print(f"  Is Connected: {'✅ YES' if graph_val['is_connected'] else '❌ NO'}")
        
        print(f"\nModel Performance:")
        for model_name, model_results in results['model_tests'].items():
            if 'error' in model_results:
                print(f"  {model_name}: ❌ ERROR - {model_results['error']}")
                continue
                
            identity_test = model_results.get('identity_comparison', {})
            diversity_test = model_results.get('spatial_diversity', {})
            
            using_graph = identity_test.get('is_using_graph', False)
            is_diverse = diversity_test.get('is_spatially_diverse', False)
            
            status = "✅ GOOD" if using_graph and is_diverse else "❌ POOR"
            print(f"  {model_name}: {status}")
            print(f"    - Using Graph: {'✅' if using_graph else '❌'}")
            print(f"    - Spatially Diverse: {'✅' if is_diverse else '❌'}")
    
    def save_results(self, results: Dict, filename: str = "graph_validation_results.json"):
        """
        Save validation results to file
        """
        import json
        
        # Convert torch tensors to Python types for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_tensors(results)
        
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")


def quick_validation_example():
    """
    Example of how to use the validator
    """
    # This is just an example - you'll need to adapt it to your actual data loading
    print("Example usage:")
    print("""
    # Load your data
    dataloader = GNNDataLoader(epidata)
    dataloader.retrieve_graph('your_graph_name')
    dataloader.construct_dataloaders(periods=8)
    
    # Create validator
    validator = GraphUtilizationValidator(dataloader)
    
    # Define models to test
    models_to_test = [
        (TemporalGCNModel, 'TemporalGCN'),
        (SpatialGCNModel, 'SpatialGCN'),
        (GATLSTMModel, 'GATLSTM'),
        (NodeLSTM, 'NodeLSTM')
    ]
    
    # Run validation
    results = validator.run_comprehensive_validation(models_to_test)
    
    # Save results
    validator.save_results(results)
    """)


if __name__ == "__main__":
    quick_validation_example()
