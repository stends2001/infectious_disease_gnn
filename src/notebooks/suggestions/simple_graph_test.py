#!/usr/bin/env python3
"""
Simple Graph Utilization Test
This version focuses on the core test without complex model initialization
"""

import sys
import os
sys.path.append('src')

def test_graph_files():
    """Test if graph files exist and are valid"""
    print("🔍 Checking graph files...")
    
    graph_dir = "data/graphs"
    if not os.path.exists(graph_dir):
        print(f"❌ Graph directory not found: {graph_dir}")
        return False
    
    graph_files = [
        "identity_graph_edge_index.pt",
        "boolean_neighbors_edge_index.pt", 
        "gravity_model_log_edge_index.pt",
        "gravity_model_top8_log_edge_index.pt"
    ]
    
    existing_files = []
    for file in graph_files:
        file_path = os.path.join(graph_dir, file)
        if os.path.exists(file_path):
            existing_files.append(file)
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
    
    return existing_files

def analyze_graph_structure(graph_name):
    """Analyze a specific graph structure"""
    print(f"\n📊 Analyzing {graph_name}...")
    
    try:
        import torch
        
        # Load edge index
        edge_path = f"data/graphs/{graph_name}_edge_index.pt"
        edge_index = torch.load(edge_path)
        
        # Basic stats
        num_nodes = torch.unique(edge_index).shape[0]
        num_edges = edge_index.shape[1]
        self_loops = (edge_index[0] == edge_index[1]).sum().item()
        
        # Check if it's an identity graph
        is_identity = self_loops == num_edges
        
        # Calculate density
        max_possible = num_nodes * (num_nodes - 1)
        density = num_edges / max_possible if max_possible > 0 else 0
        
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges: {num_edges}")
        print(f"  Self-loops: {self_loops}")
        print(f"  Density: {density:.4f}")
        print(f"  Is Identity Graph: {'❌ YES' if is_identity else '✅ NO'}")
        
        # Check for edge weights
        weight_path = f"data/graphs/{graph_name}_edge_weight.pt"
        if os.path.exists(weight_path):
            edge_weight = torch.load(weight_path)
            print(f"  Edge weights: ✅ (min={edge_weight.min():.4f}, max={edge_weight.max():.4f})")
        else:
            print(f"  Edge weights: ❌ Not found")
        
        return {
            'name': graph_name,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'is_identity': is_identity,
            'has_weights': os.path.exists(weight_path)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {graph_name}: {str(e)}")
        return None

def test_simple_model_behavior():
    """Test model behavior with simple PyTorch operations"""
    print("\n🧪 Simple Model Behavior Test")
    print("=" * 40)
    
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv
        
        # Create simple test data
        num_nodes = 10
        num_features = 5
        periods = 4
        
        x = torch.randn(num_nodes, num_features, periods)
        print(f"Test data: {x.shape[0]} nodes, {x.shape[1]} features, {x.shape[2]} periods")
        
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
        
        print(f"Graphs created: {list(graphs.keys())}")
        
        # Create a simple GCN model
        class SimpleGCN(nn.Module):
            def _init_(self, input_dim, hidden_dim, output_dim):
                super()._init_()
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, output_dim)
                self.relu = nn.ReLU()
                
            def forward(self, x, edge_index):
                # Use only the last time step
                x_t = x[:, :, -1]  # [num_nodes, num_features]
                h = self.relu(self.conv1(x_t, edge_index))
                h = self.conv2(h, edge_index)
                return h
        
        # Test the model
        model = SimpleGCN(num_features, 16, 1)
        model.eval()
        
        predictions = {}
        
        with torch.no_grad():
            for graph_name, edge_index in graphs.items():
                pred = model(x, edge_index)
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
        
        if is_using_graph:
            print("  ✅ Model is responding to different graph structures!")
        else:
            print("  ❌ Model is NOT responding to different graph structures!")
            print("  This suggests the model isn't using graph information effectively.")
        
        return {
            'is_using_graph': is_using_graph,
            'sparse_similarity': sparse_sim.item(),
            'dense_similarity': dense_sim.item(),
            'predictions': {k: v.tolist() for k, v in predictions.items()}
        }
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure you have torch-geometric installed: pip install torch-geometric")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def main():
    print("🚀 Simple Graph Structure Validation")
    print("=" * 50)
    
    # Test 1: Check if graph files exist
    existing_graphs = test_graph_files()
    
    if not existing_graphs:
        print("\n❌ No graph files found. Please generate graphs first.")
        return
    
    # Test 2: Analyze each graph
    print(f"\n📊 Analyzing {len(existing_graphs)} graphs...")
    graph_analyses = []
    
    for graph_file in existing_graphs:
        graph_name = graph_file.replace('_edge_index.pt', '')
        analysis = analyze_graph_structure(graph_name)
        if analysis:
            graph_analyses.append(analysis)
    
    # Test 3: Summary
    print(f"\n📋 Graph Summary:")
    print(f"Found {len(graph_analyses)} valid graphs:")
    
    for analysis in graph_analyses:
        status = "❌ IDENTITY" if analysis['is_identity'] else "✅ SPATIAL"
        print(f"  {analysis['name']}: {status} (density={analysis['density']:.4f})")
    
    # Test 4: Simple model behavior test
    model_result = test_simple_model_behavior()
    
    if model_result:
        print(f"\n📋 Model Test Summary:")
        if model_result['is_using_graph']:
            print("✅ Simple GCN model IS using graph structure")
        else:
            print("❌ Simple GCN model is NOT using graph structure")
            print("   This indicates a fundamental issue with graph utilization")
    
    # Test 5: Recommendations
    print(f"\n💡 Recommendations:")
    
    identity_graphs = [g for g in graph_analyses if g['is_identity']]
    spatial_graphs = [g for g in graph_analyses if not g['is_identity']]
    
    if identity_graphs:
        print(f"  ⚠️  Found {len(identity_graphs)} identity graph(s) - these should perform worst")
    
    if spatial_graphs:
        print(f"  ✅ Found {len(spatial_graphs)} spatial graph(s) - these should perform better")
        
        # Check density
        for graph in spatial_graphs:
            if graph['density'] > 0.5:
                print(f"    ⚠️  {graph['name']} is very dense (may cause over-smoothing)")
            elif graph['density'] < 0.01:
                print(f"    ⚠️  {graph['name']} is very sparse (may have disconnected components)")
            else:
                print(f"    ✅ {graph['name']} has reasonable density")
    
    # Test 6: Next steps
    print(f"\n🎯 Next Steps:")
    if model_result and model_result['is_using_graph']:
        print("✅ Basic graph utilization works - the issue is likely in your specific models")
        print("1. Check your model implementations")
        print("2. Look for spatial consistency loss that might be causing over-smoothing")
        print("3. Verify your models are actually using edge_index in forward pass")
    else:
        print("❌ Even basic graph utilization doesn't work - check your environment")
        print("1. Verify torch-geometric is properly installed")
        print("2. Check if your models are actually using graph structure")
        print("3. Look for issues in model architecture")

if _name_ == "_main_":
    main()