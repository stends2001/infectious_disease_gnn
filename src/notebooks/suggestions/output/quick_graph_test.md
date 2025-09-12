🚀 Quick Graph Structure Validation
==================================================
🔍 Checking graph files...
✅ Found: identity_graph_edge_index.pt
✅ Found: boolean_neighbors_edge_index.pt
✅ Found: gravity_model_log_edge_index.pt
✅ Found: gravity_model_top8_log_edge_index.pt

📊 Analyzing 4 graphs...

📊 Analyzing identity_graph...
/home/de-schrijvers/projects/germany_gnn/src/notebooks/suggestions/quick_graph_test.py:47: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  edge_index = torch.load(edge_path)
  Nodes: 411
  Edges: 411
  Self-loops: 411
  Density: 0.0024
  Is Identity Graph: ❌ YES
  Edge weights: ❌ Not found

📊 Analyzing boolean_neighbors...
  Nodes: 411
  Edges: 2529
  Self-loops: 411
  Density: 0.0150
  Is Identity Graph: ✅ NO
  Edge weights: ❌ Not found

📊 Analyzing gravity_model_log...
  Nodes: 411
  Edges: 4255
  Self-loops: 411
  Density: 0.0253
  Is Identity Graph: ✅ NO
/home/de-schrijvers/projects/germany_gnn/src/notebooks/suggestions/quick_graph_test.py:70: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  edge_weight = torch.load(weight_path)
  Edge weights: ✅ (min=0.0093, max=1.0000)

📊 Analyzing gravity_model_top8_log...
  Nodes: 411
  Edges: 3695
  Self-loops: 411
  Density: 0.0219
  Is Identity Graph: ✅ NO
  Edge weights: ✅ (min=0.0215, max=1.0000)

📋 Summary:
Found 4 valid graphs:
  identity_graph: ❌ IDENTITY (density=0.0024)
  boolean_neighbors: ✅ SPATIAL (density=0.0150)
  gravity_model_log: ✅ SPATIAL (density=0.0253)
  gravity_model_top8_log: ✅ SPATIAL (density=0.0219)

💡 Recommendations:
  ⚠️  Found 1 identity graph(s) - these should perform worst
  ✅ Found 3 spatial graph(s) - these should perform better
    ✅ boolean_neighbors has reasonable density
    ✅ gravity_model_log has reasonable density
    ✅ gravity_model_top8_log has reasonable density

🎯 Next Steps:
1. Run the full validation script: python validate_graph_usage.py
2. Or manually test models on different graphs
3. Check if models perform differently on identity vs spatial graphs
4. If they perform similarly, your models aren't using graph structure