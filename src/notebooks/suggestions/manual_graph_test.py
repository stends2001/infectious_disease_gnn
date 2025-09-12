#!/usr/bin/env python3
"""
Manual Graph Utilization Test
This script helps you manually test if your models are using graph structure
"""

import sys
import os
# sys.path.append('src')

def create_test_data():
    """Create simple test data for validation"""
    import torch
    import numpy as np
    
    # Create simple test data
    num_nodes = 10
    num_features = 5
    periods = 4
    
    # Random features
    x = torch.randn(num_nodes, num_features, periods)
    
    # Create different graph types
    graphs = {}
    
    # 1. Identity graph (self-loops only)
    identity_edge_index = torch.stack([
        torch.arange(num_nodes),
        torch.arange(num_nodes)
    ])
    graphs['identity'] = identity_edge_index
    
    # 2. Sparse graph (each node connects to 2 neighbors)
    sparse_edges = []
    for i in range(num_nodes):
        # Connect to next 2 nodes (with wraparound)
        for j in range(1, 3):
            neighbor = (i + j) % num_nodes
            sparse_edges.append([i, neighbor])
            sparse_edges.append([neighbor, i])  # Undirected
    
    sparse_edge_index = torch.tensor(sparse_edges).t()
    graphs['sparse'] = sparse_edge_index
    
    # 3. Dense graph (everyone connected to everyone)
    dense_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dense_edges.append([i, j])
    
    dense_edge_index = torch.tensor(dense_edges).t()
    graphs['dense'] = dense_edge_index
    
    return x, graphs

def test_model_behavior(model_class, model_name, x, graphs):
    """Test how a model behaves with different graphs"""
    print(f"\n🧪 Testing {model_name}")
    print("-" * 30)
    
    try:
        # Create a simple dataloader mock
        class MockDataLoader:
            def __init__(self):
                self.feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
                self.periods = 4
                self.prediction_horizon = 1
        
        dataloader = MockDataLoader()
        
        # Initialize model
        model = model_class(dataloader, name=model_name)
        model.set_model_hparams()
        model.model.eval()
        
        predictions = {}
        
        with torch.no_grad():
            for graph_name, edge_index in graphs.items():
                pred = model.model(x, edge_index)
                predictions[graph_name] = pred
                print(f"  {graph_name}: mean={pred.mean():.4f}, std={pred.std():.4f}")
        
        # Compare predictions
        identity_pred = predictions['identity']
        sparse_pred = predictions['sparse']
        dense_pred = predictions['dense']
        
        # Calculate similarities
        sparse_sim = torch.cosine_similarity(identity_pred, sparse_pred, dim=0)
        dense_sim = torch.cosine_similarity(identity_pred, dense_pred, dim=0)
        
        print(f"  Similarity (identity vs sparse): {sparse_sim:.4f}")
        print(f"  Similarity (identity vs dense): {dense_sim:.4f}")
        
        # Determine if model is using graph structure
        is_using_graph = sparse_sim < 0.9 or dense_sim < 0.9
        
        print(f"  Using Graph Structure: {'✅ YES' if is_using_graph else '❌ NO'}")
        
        return {
            'model_name': model_name,
            'is_using_graph': is_using_graph,
            'sparse_similarity': sparse_sim.item(),
            'dense_similarity': dense_sim.item(),
            'predictions': {k: v.tolist() for k, v in predictions.items()}
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return {'model_name': model_name, 'error': str(e)}

def main():
    print("🚀 Manual Graph Utilization Test")
    print("=" * 50)
    
    try:
        import torch
        from ...models.temporal_gcn import TemporalGCNModel
        from ...models.spatial_gcn import SpatialGCNModel
        from ...models.gnnlstmgat import GATLSTMModel
        
        # Create test data
        print("📊 Creating test data...")
        x, graphs = create_test_data()
        
        print(f"Test data: {x.shape[0]} nodes, {x.shape[1]} features, {x.shape[2]} periods")
        print(f"Graphs: {list(graphs.keys())}")
        
        # Test models
        models_to_test = [
            (TemporalGCNModel, 'TemporalGCN'),
            (SpatialGCNModel, 'SpatialGCN'),
            (GATLSTMModel, 'GATLSTM')
        ]
        
        results = []
        
        for model_class, model_name in models_to_test:
            result = test_model_behavior(model_class, model_name, x, graphs)
            results.append(result)
        
        # Summary
        print(f"\n📋 Summary:")
        print("=" * 30)
        
        for result in results:
            if 'error' in result:
                print(f"{result['model_name']}: ❌ ERROR")
            else:
                status = "✅ Using graph" if result['is_using_graph'] else "❌ Not using graph"
                print(f"{result['model_name']}: {status}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        using_graph_models = [r for r in results if r.get('is_using_graph', False)]
        not_using_graph_models = [r for r in results if not r.get('is_using_graph', False) and 'error' not in r]
        
        if using_graph_models:
            print(f"✅ {len(using_graph_models)} model(s) are using graph structure")
        
        if not_using_graph_models:
            print(f"❌ {len(not_using_graph_models)} model(s) are NOT using graph structure")
            print("   These models need to be fixed!")
        
        if not using_graph_models and not not_using_graph_models:
            print("⚠️  All models had errors - check your model implementations")
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure you have all required dependencies installed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
