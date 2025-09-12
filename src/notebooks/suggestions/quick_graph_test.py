#!/usr/bin/env python3
"""
Quick and simple graph utilization test
Run this to immediately check if your models are using graph structure
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

def simple_model_test():
    """Simple test to see if models behave differently"""
    print("\n🧪 Simple Model Behavior Test")
    print("=" * 40)
    
    try:
        # This is a minimal test - you'll need to adapt it to your actual setup
        print("To run a full model test, you need to:")
        print("1. Load your data with EpiDataLoader")
        print("2. Create GNNDataLoader")
        print("3. Load different graphs")
        print("4. Test models on each graph")
        print("5. Compare predictions")
        
        print("\n💡 Quick manual test you can do:")
        print("1. Train a model on identity_graph")
        print("2. Train the same model on boolean_neighbors")
        print("3. Compare their predictions")
        print("4. If they're similar, the model isn't using graph structure")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    print("🚀 Quick Graph Structure Validation")
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
    print(f"\n📋 Summary:")
    print(f"Found {len(graph_analyses)} valid graphs:")
    
    for analysis in graph_analyses:
        status = "❌ IDENTITY" if analysis['is_identity'] else "✅ SPATIAL"
        print(f"  {analysis['name']}: {status} (density={analysis['density']:.4f})")
    
    # Test 4: Recommendations
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
    
    # Test 5: Next steps
    print(f"\n🎯 Next Steps:")
    print(f"1. Run the full validation script: python validate_graph_usage.py")
    print(f"2. Or manually test models on different graphs")
    print(f"3. Check if models perform differently on identity vs spatial graphs")
    print(f"4. If they perform similarly, your models aren't using graph structure")

if __name__ == "__main__":
    main()