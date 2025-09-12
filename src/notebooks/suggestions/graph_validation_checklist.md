# Graph Structure Validation Checklist

## Quick Tests You Can Run Right Now

### 1. **Check Your Graph Files** (2 minutes)
```bash
python quick_graph_test.py
```
This will:
- ✅ Verify graph files exist
- ✅ Check if identity graph is actually identity
- ✅ Analyze graph density
- ✅ Identify potential issues

### 2. **Manual Model Test** (5 minutes)
```bash
python manual_graph_test.py
```
This will:
- ✅ Test models with different graph types
- ✅ Compare predictions across graphs
- ✅ Identify which models use graph structure

### 3. **Full Validation** (10 minutes)
```bash
python validate_graph_usage.py
```
This will:
- ✅ Run comprehensive tests
- ✅ Test all your models
- ✅ Generate detailed reports

## What to Look For

### ✅ **Good Signs (Models ARE using graph structure)**
- Identity graph predictions are different from other graphs
- Cosine similarity < 0.9 between identity and spatial graphs
- Models show spatial diversity in predictions
- Different models behave differently on same graph

### ❌ **Bad Signs (Models are NOT using graph structure)**
- All graphs produce similar predictions
- Cosine similarity > 0.9 between identity and spatial graphs
- Predictions are too smooth/similar across nodes
- All models behave identically

## Expected Results

### **Identity Graph**
- Should perform worst
- Should produce different predictions than other graphs
- If models perform similarly on identity vs other graphs → **PROBLEM**

### **Boolean Neighbors Graph**
- Should be sparse but meaningful
- Should show local spatial patterns
- Should perform better than identity graph

### **Gravity Model Graph**
- Should be denser but weighted
- Should capture population-based interactions
- Should perform best (if properly constructed)

## Quick Fixes to Try

### 1. **Remove Spatial Consistency Loss**
```python
# In your training loop, replace:
loss = spatial_consistency_loss(y_hat, snapshot.y, snapshot.edge_index)

# With:
loss = spike_weighted_mse(y_hat, snapshot.y)
```

### 2. **Create Sparse Graphs**
```python
# In graphconstructor.py, use:
graph_constructor.generate_graph(
    method='k_nearest', 
    k=5,  # Instead of dense connections
    name_addition='sparse'
)
```

### 3. **Add Graph Validation**
```python
# Add this to your model training:
def validate_graph_usage(model, test_sample):
    # Test with identity graph
    identity_edge_index = torch.stack([
        torch.arange(test_sample.x.shape[0]),
        torch.arange(test_sample.x.shape[0])
    ])
    
    pred_actual = model(test_sample.x, test_sample.edge_index)
    pred_identity = model(test_sample.x, identity_edge_index)
    
    similarity = torch.cosine_similarity(pred_actual, pred_identity, dim=0)
    print(f"Graph utilization: {'GOOD' if similarity < 0.9 else 'POOR'}")
```

## Troubleshooting

### **Problem: All models perform similarly**
**Solution**: Check if you're using spatial consistency loss - remove it!

### **Problem: Models perform well on identity graph**
**Solution**: Your models aren't using graph structure - check model architecture

### **Problem: Predictions are too smooth**
**Solution**: Graph is too dense - create sparser graphs

### **Problem: Predictions are too noisy**
**Solution**: Graph is too sparse - increase connectivity

## Success Criteria

You'll know you've fixed the problem when:

1. ✅ **Identity graph performs worst** (as expected)
2. ✅ **Different graphs produce different results**
3. ✅ **Models show spatial diversity** in predictions
4. ✅ **tGCN performs better** than other models (it has both spatial + temporal)
5. ✅ **NodeLSTM performs worst** (it ignores spatial structure)

## Next Steps After Validation

1. **If models ARE using graph structure**: Great! Focus on improving performance
2. **If models are NOT using graph structure**: Fix the issues identified in suggestions.md
3. **If some models work, others don't**: Fix the broken models first

## Files Created for You

- `quick_graph_test.py` - Quick graph file validation
- `manual_graph_test.py` - Simple model behavior test  
- `validate_graph_usage.py` - Full validation framework
- `src/validation/graph_utilization_validator.py` - Comprehensive validator class

Run these in order to diagnose your graph utilization issues!
