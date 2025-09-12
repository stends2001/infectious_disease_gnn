# Spatio-temporal Modelling of Infectious Diseases in Germany

## Abstract

Infectious diseases remain a significant threat to public health worldwide, with their spread often exhibiting complex spatial and temporal patterns influenced by various regional factors. Recently, Graph Neural Networks (GNNs) have emerged as a powerful tool for modeling structured data, making them well-suited for capturing the intricate relationships inherent in epidemiological data across interconnected regions.

This project aims to develop a comprehensive pipeline for applying GNN-based models to regional infectious disease prediction across multiple diseases in Germany, using data extracted from SurvStat. The pipeline will streamline data processing, model training, evaluation, and interpretation, providing a flexible framework adaptable to different diseases and regional settings. By leveraging GNNs' ability to integrate spatial connectivity and temporal dynamics, the project seeks to improve forecasting accuracy and support data-driven public health decision-making.

Ultimately, this pipeline will facilitate systematic experimentation with various graph architectures and epidemiological datasets, advancing the understanding of how spatial dependencies and regional interactions drive disease spread.

## Project Goals and Aims

### Primary Objectives
1. **Develop a comprehensive GNN-based pipeline** for infectious disease forecasting in Germany
2. **Integrate spatial and temporal dependencies** to improve prediction accuracy
3. **Create flexible graph construction methods** that capture disease transmission patterns
4. **Benchmark different GNN architectures** against traditional epidemiological models
5. **Provide interpretable results** for public health decision-making

### Research Questions
- How do different graph structures affect disease prediction accuracy?
- Which GNN architectures best capture spatio-temporal disease dynamics?
- How can we optimize graph density for disease transmission modeling?
- What are the trade-offs between model complexity and prediction performance?

## Methods

### Data Sources
- **SurvStat**: German infectious disease surveillance data
- **Geographic Data**: Shapefiles for German administrative regions (Kreise level)
- **Population Data**: Regional population statistics for weighting connections

### Graph Construction Methods

#### 1. Boolean Neighbors Graph
- **Method**: Connects geographically adjacent regions (touching boundaries)
- **Use Case**: Captures direct spatial adjacency
- **Characteristics**: Sparse, preserves local connectivity

#### 2. Identity Graph
- **Method**: Each node only connects to itself (self-loops only)
- **Use Case**: Baseline comparison to test if models use graph structure
- **Characteristics**: Minimal connectivity, should perform worst

#### 3. Mesh Graph
- **Method**: Connects every node to every other node
- **Use Case**: Maximum connectivity baseline
- **Characteristics**: Dense, may cause over-smoothing

#### 4. Distance Threshold Graph
- **Method**: Connects nodes within a specified distance threshold
- **Use Case**: Captures regional influence based on proximity
- **Characteristics**: Density controlled by distance parameter

#### 5. K-Nearest Neighbors Graph
- **Method**: Each node connects to its k nearest neighbors
- **Use Case**: Balanced connectivity with controlled density
- **Characteristics**: Uniform degree distribution

#### 6. Population Weighted Graph
- **Method**: Connections weighted by population size and distance
- **Use Case**: Reflects human mobility patterns
- **Characteristics**: Weighted edges, population-dependent

#### 7. Gravity Model Graph
- **Method**: Connections based on gravity model: `weight = (pop_i * pop_j) / distance^α`
- **Use Case**: Models human interaction patterns
- **Characteristics**: Realistic transmission modeling, density control options

### Model Architectures

#### 1. GAT-LSTM Model (`gnnlstmgat.py`)
- **Architecture**: Graph Attention Network + LSTM
- **Approach**: 
  - Applies GAT at each time step independently
  - Uses LSTM for temporal sequence modeling
  - Combines spatial and temporal processing
- **Strengths**: Attention mechanism, temporal modeling
- **Limitations**: Static graph structure, no temporal graph evolution

#### 2. Spatial GCN Model (`spatial_gcn.py`)
- **Architecture**: Pure Graph Convolutional Network
- **Approach**:
  - Only uses most recent time step
  - Focuses purely on spatial relationships
  - Multiple GCN layers with dropout
- **Strengths**: Simple, interpretable spatial processing
- **Limitations**: Ignores temporal dimension completely

#### 3. Temporal GCN Model (`temporal_gcn.py`)
- **Architecture**: GCN + LSTM hybrid
- **Approach**:
  - Applies GCN at each time step
  - Uses LSTM for temporal sequence modeling
  - Processes spatial and temporal dimensions
- **Strengths**: Balanced spatial-temporal processing
- **Limitations**: Independent time step processing

#### 4. A3TGCN Model (`a3tgcn.py`)
- **Architecture**: Adaptive Adjacency Attention Temporal GCN
- **Approach**: Advanced temporal GCN with attention mechanisms
- **Strengths**: State-of-the-art temporal GNN architecture
- **Limitations**: More complex, requires careful tuning

#### 5. Node LSTM Model (`node_lstm.py`)
- **Architecture**: Independent LSTM per node
- **Approach**: Treats each region independently
- **Strengths**: Simple baseline, no graph assumptions
- **Limitations**: Ignores spatial relationships completely

### Loss Functions

#### 1. Standard MSE
- **Purpose**: Basic regression loss
- **Formula**: `MSE = mean((y_pred - y_true)²)`

#### 2. Spike Weighted MSE
- **Purpose**: Emphasize high-magnitude predictions
- **Formula**: `Loss = mean((1 + |y_true|) * (y_pred - y_true)²)`
- **Use Case**: Disease outbreaks and spikes

#### 3. Spike Timing Weighted MSE
- **Purpose**: Emphasize temporal changes and spikes
- **Formula**: `Loss = mean(base_weight * (1 + 5 * |dy|) * (y_pred - y_true)²)`
- **Use Case**: Capturing rapid changes in disease incidence

#### 4. Spatial Consistency Loss
- **Purpose**: Encourage similar predictions for neighboring nodes
- **Formula**: `Loss = MSE + α * mean((pred_i - pred_j)²)`
- **Use Case**: Spatial smoothing (may be counterproductive for disease modeling)

#### 5. Spike Detection Loss
- **Purpose**: Different treatment for spike vs. normal periods
- **Formula**: `Loss = 2.0 * spike_loss + normal_loss`
- **Use Case**: Binary classification of outbreak periods

### Data Processing Pipeline

#### 1. Data Loading (`epidataloader.py`)
- Loads infectious disease data from SurvStat
- Handles multiple diseases (COVID-19, Influenza)
- Manages temporal and spatial data organization

#### 2. Feature Engineering
- **Time Features**: Seasonal patterns, day of year, etc.
- **Lag Features**: Historical values for autoregressive modeling
- **Normalization**: Z-score normalization for stable training
- **Log Transformation**: Handles skewed disease incidence data

#### 3. Graph Construction (`graphconstructor.py`)
- Multiple graph construction algorithms
- Distance and population-based weighting
- Density control mechanisms
- Graph visualization and validation

#### 4. GNN Data Loading (`gnndataloader.py`)
- Converts tabular data to graph format
- Creates temporal sequences for GNN training
- Manages train/validation/test splits
- Handles edge indices and weights

### Evaluation Metrics

#### 1. Standard Regression Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

#### 2. Disease-Specific Metrics
- **Spike Detection**: Precision, Recall, F1-score for outbreak detection
- **Spatial Diversity**: Variance in predictions across regions
- **Temporal Accuracy**: Performance on different time horizons

#### 3. Model Comparison Metrics
- **Graph Utilization**: Test if models actually use graph structure
- **Baseline Comparison**: Performance vs. simple baselines
- **Cross-Graph Performance**: Consistency across different graph types

## Current Issues and Challenges

### 1. Graph Structure Utilization
- **Problem**: Models perform similarly across different graph types
- **Evidence**: Identity graph performs similarly to other graphs
- **Impact**: Models not leveraging spatial information effectively

### 2. Over-Smoothing
- **Problem**: Predictions are too smooth and similar across nodes
- **Causes**: Dense graphs, inappropriate loss functions
- **Impact**: Loss of local disease patterns

### 3. Loss Function Design
- **Problem**: Spatial consistency loss encourages smoothing
- **Impact**: Counterproductive for disease spread modeling
- **Solution**: Need spatial diversity loss instead

### 4. Graph Density Control
- **Problem**: Graphs either too sparse or too dense
- **Impact**: Poor message passing or over-smoothing
- **Solution**: Need adaptive density control

## To Dos

### Immediate Priorities
- [ ] Fix spatial consistency loss function
- [ ] Add graph utilization validation
- [ ] Create less dense graphs (k=5-8 neighbors)
- [ ] Implement spatial diversity loss

### Short Term Goals
- [ ] Add edge weight support to all models
- [ ] Create distance-banded graphs
- [ ] Implement graph structure validation
- [ ] Add model comparison framework

### Long Term Objectives
- [ ] Implement adaptive graph construction
- [ ] Add temporal graph evolution
- [ ] Create disease-specific graph patterns
- [ ] Develop interpretability tools

## Literature Review

### Key References
- **Kraemer, 2024**: GNN applications in epidemiology
- **Croft, 2023**: Spatial-temporal disease modeling
- **Jeong, 2025**: Advanced GNN architectures for disease prediction
- **Wang, 2025**: SIR-based models for multi-region prediction
- **Liu, 2024**: GRGNN (GRU-based GNN) for disease forecasting

### Methodological Foundations
- **Graph Neural Networks**: Spatial information processing
- **Temporal Modeling**: LSTM, GRU, and temporal convolutions
- **Epidemiological Modeling**: SIR models, gravity models
- **Spatial Analysis**: Geographic information systems, spatial statistics

## Technical Implementation

### Dependencies
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural network library
- **GeoPandas**: Geographic data processing
- **Scikit-learn**: Machine learning utilities
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

### Project Structure
```
src/
├── dataloading/          # Data processing and graph construction
├── models/              # GNN model implementations
├── metrics/             # Loss functions and evaluation metrics
└── notebooks/           # Analysis and experimentation notebooks

data/
├── graphs/              # Pre-computed graph structures
├── population/          # Population data
└── shape/              # Geographic shapefiles
```

### Usage Example
```python
# Load data
epidata = EpiDataLoader('influenza', data_env, aggr_level='03')
epidata.add_time_features()
epidata.log_transform_target()
epidata.normalize('2018-06-01', '2019-06-01', 'zscore')

# Create graph
graph_constructor = GraphConstructor(
    graph_dir='data/graphs',
    population_data=epidata.population_by_node,
    shapes=epidata.shapedata,
    id_col='node'
)
graph_constructor.generate_graph(method='gravity_model', k=8)
graph_constructor.save_graph('gravity_model_sparse')

# Train model
dataloader = GNNDataLoader(epidata)
dataloader.retrieve_graph('gravity_model_sparse')
dataloader.construct_dataloaders(periods=8)

model = TemporalGCNModel(dataloader)
model.set_model_hparams(hidden_size=64, num_layers=2)
model.train(n_epochs=100)
model.forecast()
```

## Future Directions

### 1. Advanced Graph Construction
- Dynamic graphs that evolve over time
- Disease-specific transmission patterns
- Multi-scale graph hierarchies

### 2. Model Improvements
- Attention mechanisms for temporal graphs
- Graph neural ODEs for continuous-time modeling
- Multi-task learning for different diseases

### 3. Interpretability
- Graph attention visualization
- Spatial influence analysis
- Temporal pattern identification

### 4. Real-world Deployment
- Real-time prediction systems
- Integration with public health databases
- Automated alert systems for disease outbreaks

This comprehensive framework provides a solid foundation for spatio-temporal disease modeling using graph neural networks, with clear methodologies, implementations, and future research directions.
