# GNN Performance Analysis and Suggestions

## Problem Summary

Your GNNs are performing very similarly across different graph structures, including the identity graph (where nodes only connect to themselves). This suggests that the models are not properly utilizing the graph structure and are essentially learning node-independent patterns, leading to smooth, similar predictions across all nodes.

## Root Cause Analysis

### 1. **Graph Structure Issues**

#### Identity Graph Problem
- **Issue**: The identity graph only contains self-loops, meaning each node only connects to itself
- **Impact**: GNN layers become essentially identity transformations, making the graph structure meaningless
- **Evidence**: Your models perform similarly on identity graphs vs. other graphs, which shouldn't happen if graph structure is being used

#### Graph Density Issues
Based on the graph construction methods in your code:

1. **Boolean Neighbors**: Only connects geographically adjacent regions
   - **Problem**: May create disconnected components
   - **Density**: Likely very sparse, potentially too sparse for effective message passing

2. **Gravity Model**: Connects all regions within distance threshold
   - **Problem**: Can be extremely dense, leading to over-smoothing
   - **Density**: May connect too many nodes, diluting local patterns

3. **Mesh Graph**: Connects every node to every other node
   - **Problem**: Maximum density, complete over-smoothing
   - **Impact**: All nodes receive identical information from all other nodes

### 2. **Model Architecture Issues**

#### GAT-LSTM Model (`gnnlstmgat.py`)
```python
# Line 40: GAT applied per time step
xt_gat = self.gat(xt, edge_index)
```
- **Issue**: GAT is applied independently at each time step
- **Problem**: No temporal graph evolution, static graph structure
- **Impact**: Temporal dependencies are handled by LSTM, not by graph structure

#### Spatial GCN Model (`spatial_gcn.py`)
```python
# Line 53: Only uses most recent time step
x_t = x[:, :, -1]  # [num_nodes, node_features]
```
- **Issue**: Completely ignores temporal dimension
- **Problem**: Only uses graph structure at the final time step
- **Impact**: No temporal-spatial interaction

#### Temporal GCN Model (`temporal_gcn.py`)
```python
# Lines 69-80: Spatial processing per time step
for t in range(time_steps):
    x_t = x[:, :, t]
    # Apply spatial GCN layers
```
- **Issue**: Better approach but still processes time steps independently
- **Problem**: No cross-temporal graph message passing

### 3. **Loss Function Issues**

#### Spike Weighted MSE (`losses.py` line 4-6)
```python
def spike_weighted_mse(y_pred, y_true):
    weights = 1 + torch.abs(y_true)  # weight more on higher target magnitude
    return torch.mean(weights * (y_pred - y_true)**2)
```
- **Issue**: Weights higher values more, but doesn't encourage spatial diversity
- **Problem**: May encourage similar predictions across nodes
- **Impact**: Loss doesn't penalize over-smoothing

#### Spatial Consistency Loss (`losses.py` lines 64-90)
```python
def spatial_consistency_loss(y_pred, y_true, edge_index, alpha=0.1):
    # Penalizes predictions that don't respect spatial relationships
    spatial_loss = torch.mean((pred_i - pred_j)**2)
    return mse_loss + alpha * spatial_loss
```
- **Issue**: Encourages neighboring nodes to have similar predictions
- **Problem**: This is the opposite of what you want for disease spread
- **Impact**: Forces smooth predictions, reducing spatial diversity

## Detailed Suggestions

### 1. **Fix Graph Construction**

#### Create Less Dense Graphs
```python
# Add to graphconstructor.py
def generate_sparse_gravity_model(df, population_data, max_distance, 
                                 id_col='id', k=5, weight_threshold=0.1):
    """
    Create a sparser gravity model with better density control
    """
    # ... existing gravity model code ...
    
    # Apply stricter density control
    if density_control == 'adaptive_k':
        # Use adaptive k based on population density
        for i in range(len(df)):
            pop_density = df.iloc[i]['population_size'] / df.iloc[i]['geometry'].area
            adaptive_k = max(3, min(k, int(pop_density * 0.1)))
            # ... rest of logic
```

#### Distance-Based Sparse Graphs
```python
def generate_distance_band_graph(df, distance_bands, id_col='id'):
    """
    Create graphs with specific distance bands
    distance_bands: [(min_dist, max_dist, weight), ...]
    """
    edges = []
    weights = []
    
    for min_dist, max_dist, weight in distance_bands:
        # Connect nodes within this distance band
        # This creates more controlled connectivity
```

#### K-Nearest Neighbors with Distance Threshold
```python
def generate_knn_with_threshold(df, k=8, max_distance=200000, id_col='id'):
    """
    Combine k-nearest neighbors with distance threshold
    """
    # Ensures each node has exactly k neighbors (if within distance)
    # But doesn't connect nodes that are too far apart
```

### 2. **Improve Model Architectures**

#### Fix GAT-LSTM to Use Edge Weights
```python
# In gnnlstmgat.py, modify forward method:
def forward(self, x, edge_index, edge_weight=None):
    gatt_out_seq = []
    
    for t in range(self.periods):
        xt = x[:, :, t]
        # Use edge weights if available
        if edge_weight is not None:
            xt_gat = self.gat(xt, edge_index, edge_weight)
        else:
            xt_gat = self.gat(xt, edge_index)
        xt_gat = torch.relu(xt_gat)
        gatt_out_seq.append(xt_gat)
    
    # ... rest of method
```

#### Add Graph Structure Validation
```python
def validate_graph_structure(self, edge_index, edge_weight=None):
    """
    Validate that graph structure is being used effectively
    """
    num_nodes = torch.unique(edge_index).shape[0]
    num_edges = edge_index.shape[1]
    
    # Check for identity graph
    self_loops = (edge_index[0] == edge_index[1]).sum().item()
    if self_loops == num_edges:
        print("WARNING: Identity graph detected - no spatial information!")
        return False
    
    # Check density
    max_possible = num_nodes * (num_nodes - 1)
    density = num_edges / max_possible
    if density > 0.5:
        print(f"WARNING: Very dense graph (density={density:.3f}) - may cause over-smoothing")
    elif density < 0.01:
        print(f"WARNING: Very sparse graph (density={density:.3f}) - may have disconnected components")
    
    return True
```

### 3. **Fix Loss Functions**

#### Remove Spatial Consistency Loss
The current spatial consistency loss is counterproductive for disease modeling:
```python
# REMOVE or modify this loss function
def spatial_consistency_loss(y_pred, y_true, edge_index, alpha=0.1):
    # This encourages smooth predictions - BAD for disease spread
    # Instead, use spatial diversity loss
```

#### Add Spatial Diversity Loss
```python
def spatial_diversity_loss(y_pred, y_true, edge_index, alpha=0.1):
    """
    Encourage spatial diversity in predictions
    """
    mse_loss = torch.mean((y_pred - y_true)**2)
    
    # Encourage different predictions for different nodes
    node_variance = torch.var(y_pred)
    diversity_loss = -node_variance  # Negative because we want high variance
    
    return mse_loss + alpha * diversity_loss
```

#### Add Spike-Aware Spatial Loss
```python
def spike_aware_spatial_loss(y_pred, y_true, edge_index, alpha=0.1):
    """
    Loss that encourages spatial patterns while preserving spikes
    """
    mse_loss = torch.mean((y_pred - y_true)**2)
    
    # Only apply spatial smoothing to non-spike regions
    spike_threshold = torch.quantile(y_true, 0.8)  # Top 20% are spikes
    spike_mask = y_true > spike_threshold
    
    if edge_index.shape[1] > 0:
        pred_i = y_pred[edge_index[0]]
        pred_j = y_pred[edge_index[1]]
        
        # Only smooth non-spike regions
        non_spike_i = ~spike_mask[edge_index[0]]
        non_spike_j = ~spike_mask[edge_index[1]]
        non_spike_edges = non_spike_i & non_spike_j
        
        if non_spike_edges.sum() > 0:
            spatial_loss = torch.mean((pred_i[non_spike_edges] - pred_j[non_spike_edges])**2)
        else:
            spatial_loss = 0
    else:
        spatial_loss = 0
    
    return mse_loss + alpha * spatial_loss
```

### 4. **Add Model Validation**

#### Graph Utilization Test
```python
def test_graph_utilization(model, dataloader, graph_name):
    """
    Test if model is actually using graph structure
    """
    # Test 1: Compare predictions with and without graph
    model.eval()
    
    # Get a test sample
    test_sample = dataloader.dataset_test[0]
    
    # Prediction with graph
    pred_with_graph = model(test_sample.x, test_sample.edge_index)
    
    # Create identity graph (self-loops only)
    num_nodes = test_sample.x.shape[0]
    identity_edge_index = torch.stack([
        torch.arange(num_nodes),
        torch.arange(num_nodes)
    ])
    
    # Prediction with identity graph
    pred_identity = model(test_sample.x, identity_edge_index)
    
    # Compare predictions
    similarity = torch.cosine_similarity(pred_with_graph, pred_identity, dim=0)
    
    print(f"Graph utilization test for {graph_name}:")
    print(f"  Cosine similarity with identity graph: {similarity:.4f}")
    print(f"  {'PASS' if similarity < 0.9 else 'FAIL'}: Model {'is' if similarity < 0.9 else 'is NOT'} using graph structure")
    
    return similarity < 0.9
```

### 5. **Recommended Graph Types for Disease Modeling**

#### 1. **Distance-Banded Graph**
```python
# Connect nodes in specific distance bands
distance_bands = [
    (0, 50000, 1.0),      # Very close neighbors
    (50000, 150000, 0.5), # Medium distance
    (150000, 300000, 0.1) # Far neighbors
]
```

#### 2. **Population-Weighted K-NN**
```python
# Each node connects to k most influential neighbors
# Weight by population and distance
k = 6  # Not too sparse, not too dense
```

#### 3. **Adaptive Density Graph**
```python
# Dense connections in urban areas, sparse in rural
# Based on population density and disease transmission patterns
```

### 6. **Implementation Priority**

1. **Immediate (High Priority)**:
   - Remove or fix spatial consistency loss
   - Add graph utilization validation
   - Create less dense graphs (k=5-8 neighbors)

2. **Short Term (Medium Priority)**:
   - Implement spatial diversity loss
   - Add edge weight support to all models
   - Create distance-banded graphs

3. **Long Term (Low Priority)**:
   - Implement adaptive graph construction
   - Add temporal graph evolution
   - Create disease-specific graph patterns

### 7. **Expected Outcomes**

After implementing these changes:
- Models should show different performance across graph types
- Identity graph should perform worst (as expected)
- Spatial diversity should increase in predictions
- Models should capture local disease patterns better
- Over-smoothing should be reduced

### 8. **Validation Strategy**

1. **Graph Structure Validation**:
   - Check graph density and connectivity
   - Verify non-identity graphs have meaningful connections
   - Test with different k values (3, 5, 8, 10)

2. **Model Behavior Validation**:
   - Compare predictions across different graph types
   - Check for spatial diversity in predictions
   - Validate that identity graph performs worst

3. **Performance Validation**:
   - Monitor loss curves for different graph types
   - Check prediction variance across nodes
   - Validate that models learn different patterns for different graphs

This analysis should help you identify why your GNNs are not utilizing graph structure effectively and provide concrete steps to fix the issues.
