from src.dataloading.deepdataloader import GNNDataLoader
from typing import Dict 
import torch
from src.models.modelcore import DeepLearningModelCore
import os
import json

class GraphUsageValidator:
    """
    Comprehensive validation framework for testing graph structure utilization
    """
    
    def __init__(self, model: DeepLearningModelCore):
        self.model = model
        self.logs  = {}    

    def test1(self):
        """
        Test 1: Compare predictions with identity graph vs. actual graph
        This is the most important test - models should behave differently
        """
        model      = self.model
        dataloader = self.model.dataloader
        # print(f"\n🔍 Testing {model.name} - Identity Graph Comparison")
        
        # Get test sample
        test_sample = dataloader.dataset_test[10].to(model.device)
        num_nodes   = test_sample.x.shape[0]
        
        # Create identity graph (self-loops only)
        identity_edge_index = torch.stack([
            torch.arange(num_nodes),
            torch.arange(num_nodes)
        ]).to(model.device)
        
        with torch.no_grad():
            # Prediction with actual graph
            pred_actual = model.model(test_sample.x, test_sample.edge_index, test_sample.edge_weight)
            
            # Prediction with identity graph
            pred_identity = model.model(test_sample.x, identity_edge_index, None)
        
        if model.dataloader.prediction_horizon > 1:

            pred_actual = pred_actual[:,0]
            pred_identity = pred_identity[:,0]    

        # Calculate similarity metrics
        cosine_sim  = torch.cosine_similarity(pred_actual, pred_identity, dim=0)
        mse_diff    = torch.mean((pred_actual - pred_identity)**2)
        correlation = torch.corrcoef(torch.stack([pred_actual, pred_identity]))[0, 1]
        
        # Determine if model is using graph structure
        is_using_graph = cosine_sim < 0.9 and mse_diff > 0.01
        
        results = {
            'model_name': model.name,
            'cosine_similarity': cosine_sim.item(),
            'mse_difference': mse_diff.item(),
            'correlation': correlation.item(),
            'is_using_graph': is_using_graph,
            'pred_actual_mean': pred_actual.mean().item(),
            'pred_identity_mean': pred_identity.mean().item(),
            'pred_actual_std': pred_actual.std().item(),
            'pred_identity_std': pred_identity.std().item()
        }
        
        # print(f"  Cosine Similarity: {cosine_sim:.4f}")
        # print(f"  MSE Difference: {mse_diff:.4f}")
        # print(f"  Correlation: {correlation:.4f}")
        # print(f"  Using Graph Structure: {'✅ YES' if is_using_graph else '❌ NO'}")
        
        self.logs['test1'] = results 

    def test2(self):     

        """
        Test 2: Check if model is permutation invariant
        Predictions should change when node order changes (if using graph structure)
        """
        
        model      = self.model
        dataloader = self.model.dataloader
        # print(f"\n🔍 Testing {model.name} - Graph Permutation Invariance")
        test_sample = dataloader.dataset_test[10]
        
        # Create random permutation
        num_nodes = test_sample.x.shape[0]
        perm = torch.randperm(num_nodes)
        
        # Permute features and edge indices
        x_perm = test_sample.x[perm]
        edge_index_perm = torch.zeros_like(test_sample.edge_index)
        for i in range(test_sample.edge_index.shape[1]):
            edge_index_perm[0, i] = perm[test_sample.edge_index[0, i]]
            edge_index_perm[1, i] = perm[test_sample.edge_index[1, i]]
        

        test_sample = test_sample.to(model.device)
        edge_index_perm=edge_index_perm.to(model.device)
        
        with torch.no_grad():
            # Original prediction
            pred_original = model.model(test_sample.x, test_sample.edge_index, test_sample.edge_weight)
            
            # Permuted prediction
            pred_permuted = model.model(x_perm, edge_index_perm, test_sample.edge_weight)
        
        if model.dataloader.prediction_horizon > 1:

            pred_original= pred_original[:,0]
            pred_permuted= pred_permuted[:,0]

        # Inverse permute the permuted prediction
        pred_permuted_inv = pred_permuted[torch.argsort(perm)]
        
        # Calculate similarity
        cosine_sim = torch.cosine_similarity(pred_original, pred_permuted_inv, dim=0)
        mse_diff = torch.mean((pred_original - pred_permuted_inv)**2)
        
        # For graph-aware models, predictions should be similar after inverse permutation
        is_permutation_invariant = cosine_sim > 0.95 and mse_diff < 0.01
        
        results = {
            'model_name': model.name,
            'cosine_similarity': cosine_sim.item(),
            'mse_difference': mse_diff.item(),
            'is_permutation_invariant': is_permutation_invariant
        }
        
        # print(f"  Cosine Similarity: {cosine_sim:.4f}")
        # print(f"  MSE Difference: {mse_diff:.4f}")
        # print(f"  Permutation Invariant: {'✅ YES' if is_permutation_invariant else '❌ NO'}")
        self.logs['test2'] = results
        
    def test3(self):
        """
        Test 3: Edge Modification Sensitivity
        Tests how predictions change when connections between nodes are modified.
        A graph-aware model should show significant changes when edges are altered.
        """
        model      = self.model
        dataloader = self.model.dataloader    
        print(f"\n🔍 Testing {model.name} - Edge Modification Sensitivity")
        
        test_sample = dataloader.dataset_test[10].to(model.device)
        num_nodes = test_sample.x.shape[0]
        original_edge_index = test_sample.edge_index.clone()
        original_edge_weight = test_sample.edge_weight.clone() if test_sample.edge_weight is not None else None
        
        # Get original prediction
        with torch.no_grad():
            pred_original = model.model(test_sample.x, original_edge_index, original_edge_weight)
        if model.dataloader.prediction_horizon > 1:        
            pred_original = pred_original[:, 0]
        
        # Test 1: Remove random edges (20% of edges)
        num_edges = original_edge_index.shape[1]
        num_remove = max(1, int(0.2 * num_edges))
        edges_to_remove = torch.randperm(num_edges)[:num_remove]
        
        # Create mask to keep edges (True = keep, False = remove)
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        keep_mask[edges_to_remove] = False
        
        edge_index_reduced = original_edge_index[:, keep_mask]
        edge_weight_reduced = original_edge_weight[keep_mask] if original_edge_weight is not None else None
        
        with torch.no_grad():
            pred_reduced = model.model(test_sample.x, edge_index_reduced, edge_weight_reduced)

        if model.dataloader.prediction_horizon > 1:
            pred_reduced = pred_reduced[:, 0]
        
        # Test 2: Add random edges (add 10% more edges)
        num_add = max(1, int(0.1 * num_edges))
        new_edges = []
        
        for _ in range(num_add):
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            # Avoid self-loops and existing edges
            while src == dst or _edge_exists(original_edge_index, src, dst):
                src = torch.randint(0, num_nodes, (1,)).item()
                dst = torch.randint(0, num_nodes, (1,)).item()
            new_edges.extend([[src, dst], [dst, src]])  # Add both directions for undirected
        
        if new_edges:
            new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t().to(model.device)
            edge_index_added = torch.cat([original_edge_index, new_edges_tensor], dim=1)
            
            if original_edge_weight is not None:
                # Add weights for new edges (use mean weight)
                mean_weight = original_edge_weight.mean()
                new_weights = torch.full((new_edges_tensor.shape[1],), mean_weight.item()).to(model.device)
                edge_weight_added = torch.cat([original_edge_weight, new_weights])
            else:
                edge_weight_added = None
        else:
            edge_index_added = original_edge_index
            edge_weight_added = original_edge_weight
        
        with torch.no_grad():
            pred_added = model.model(test_sample.x, edge_index_added, edge_weight_added)
        if model.dataloader.prediction_horizon > 1:
            pred_added = pred_added[:, 0]
        
        # Test 3: Rewire edges (keep same number but change connections)
        edge_index_rewired = original_edge_index.clone()
        num_rewire = max(1, int(0.15 * num_edges))
        
        for i in range(num_rewire):
            edge_idx = torch.randint(0, num_edges, (1,)).item()
            new_src = torch.randint(0, num_nodes, (1,)).item()
            new_dst = torch.randint(0, num_nodes, (1,)).item()
            
            # Avoid self-loops
            while new_src == new_dst:
                new_dst = torch.randint(0, num_nodes, (1,)).item()
                
            edge_index_rewired[0, edge_idx] = new_src
            edge_index_rewired[1, edge_idx] = new_dst
        
        with torch.no_grad():
            pred_rewired = model.model(test_sample.x, edge_index_rewired, original_edge_weight)
        if model.dataloader.prediction_horizon > 1:        
            pred_rewired = pred_rewired[:, 0]
        
        # Calculate similarity metrics for each modification
        # Edge removal
        cosine_sim_removed = torch.cosine_similarity(pred_original, pred_reduced, dim=0)
        mse_diff_removed = torch.mean((pred_original - pred_reduced)**2)
        
        # Edge addition  
        cosine_sim_added = torch.cosine_similarity(pred_original, pred_added, dim=0)
        mse_diff_added = torch.mean((pred_original - pred_added)**2)
        
        # Edge rewiring
        cosine_sim_rewired = torch.cosine_similarity(pred_original, pred_rewired, dim=0)
        mse_diff_rewired = torch.mean((pred_original - pred_rewired)**2)
        
        # Calculate average sensitivity
        avg_cosine_sim = (cosine_sim_removed + cosine_sim_added + cosine_sim_rewired) / 3
        avg_mse_diff = (mse_diff_removed + mse_diff_added + mse_diff_rewired) / 3
        
        # Determine if model is sensitive to edge modifications
        # A graph-aware model should show significant changes (lower similarity, higher MSE)
        is_edge_sensitive = avg_cosine_sim < 0.8 and avg_mse_diff > 0.02
        
        # Additional check: at least one modification should cause substantial change
        substantial_change = (cosine_sim_removed < 0.85 or cosine_sim_added < 0.85 or cosine_sim_rewired < 0.85)
        
        results = {
            'model_name': model.name,
            # Edge removal results
            'cosine_sim_removed': cosine_sim_removed.item(),
            'mse_diff_removed': mse_diff_removed.item(),
            # Edge addition results  
            'cosine_sim_added': cosine_sim_added.item(),
            'mse_diff_added': mse_diff_added.item(),
            # Edge rewiring results
            'cosine_sim_rewired': cosine_sim_rewired.item(), 
            'mse_diff_rewired': mse_diff_rewired.item(),
            # Summary metrics
            'avg_cosine_similarity': avg_cosine_sim.item(),
            'avg_mse_difference': avg_mse_diff.item(),
            'is_edge_sensitive': is_edge_sensitive,
            'substantial_change_detected': substantial_change,
            # Meta info
            'edges_removed': num_remove,
            'edges_added': len(new_edges) if new_edges else 0,
            'edges_rewired': num_rewire
        }
        
        # print(f"  Edge Removal - Cosine Sim: {cosine_sim_removed:.4f}, MSE: {mse_diff_removed:.4f}")
        # print(f"  Edge Addition - Cosine Sim: {cosine_sim_added:.4f}, MSE: {mse_diff_added:.4f}") 
        # print(f"  Edge Rewiring - Cosine Sim: {cosine_sim_rewired:.4f}, MSE: {mse_diff_rewired:.4f}")
        # print(f"  Average Similarity: {avg_cosine_sim:.4f}")
        # print(f"  Average MSE Difference: {avg_mse_diff:.4f}")
        # print(f"  Edge Sensitive: {'✅ YES' if is_edge_sensitive else '❌ NO'}")
        # print(f"  Substantial Change: {'✅ YES' if substantial_change else '❌ NO'}")
        
        self.logs['test3'] = results

    def run_tests(self,
                  logs_dir = 'tests/logs'):
        
        self.test1()
        self.test2()
        self.test3()

        file = os.path.join(logs_dir, f'graph_usage_validation_{self.model.name}.json')

        with open(file, 'w') as f:
                json.dump(tensor_to_python(self.logs), f, indent=2)     

        print('logs saved')   


def _edge_exists(edge_index: torch.Tensor, src: int, dst: int) -> bool:
    """
    Helper function to check if an edge exists between two nodes
    """
    for i in range(edge_index.shape[1]):
        if (edge_index[0, i].item() == src and edge_index[1, i].item() == dst) or \
           (edge_index[0, i].item() == dst and edge_index[1, i].item() == src):
            return True
    return False    


def tensor_to_python(obj):
    if isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(i) for i in obj]
    elif hasattr(obj, 'item'):  # tensor or scalar with item()
        return obj.item()
    else:
        return obj
