#!/usr/bin/env python3
"""
Quick script to validate graph structure usage in your GNN models
"""

import sys
import os
sys.path.append('src')

from validation.graph_utilization_validator import GraphUtilizationValidator
from dataloading.gnndataloader import GNNDataLoader
from dataloading.epidataloader import EpiDataLoader
from models.temporal_gcn import TemporalGCNModel
from models.spatial_gcn import SpatialGCNModel
from models.gnnlstmgat import GATLSTMModel
from models.node_lstm import NodeLSTM

def main():
    print("🔍 Graph Structure Validation Script")
    print("=" * 50)
    
    try:
        # Load your data (adapt these parameters to your setup)
        print("📊 Loading data...")
        
        # You'll need to adapt this to your actual data loading setup
        # This is a template - modify the parameters as needed
        epidata = EpiDataLoader(
            disease='influenza',  # or 'covid'
            data_env='your_data_env',  # adapt this
            aggr_level='03',
            min_date='2012-06-01',
            max_date='2020-06-01'
        )
        
        # Add features
        epidata.add_time_features()
        epidata.log_transform_target()
        epidata.normalize('2018-06-01', '2019-06-01', 'zscore')
        epidata.add_lagged_features(range(4, 8))
        
        # Create GNN dataloader
        dataloader = GNNDataLoader(epidata)
        
        # Load a graph (try different ones)
        graph_names = [
            'identity_graph',
            'boolean_neighbors', 
            'gravity_model_log',
            'gravity_model_top8_log'
        ]
        
        print(f"\nAvailable graph types: {graph_names}")
        print("Choose a graph to test (or press Enter for 'boolean_neighbors'):")
        
        # For automated testing, use boolean_neighbors
        graph_name = 'boolean_neighbors'  # Change this to test different graphs
        
        print(f"Using graph: {graph_name}")
        dataloader.retrieve_graph(graph_name)
        dataloader.construct_dataloaders(periods=8, prediction_horizon=1)
        
        # Create validator
        validator = GraphUtilizationValidator(dataloader)
        
        # Define models to test
        models_to_test = [
            (TemporalGCNModel, 'TemporalGCN'),
            (SpatialGCNModel, 'SpatialGCN'),
            (GATLSTMModel, 'GATLSTM'),
            (NodeLSTM, 'NodeLSTM')
        ]
        
        print(f"\n🧪 Testing {len(models_to_test)} models...")
        
        # Run validation
        results = validator.run_comprehensive_validation(models_to_test, test_sample_idx=0)
        
        # Save results
        output_file = f"validation_results_{graph_name}.json"
        validator.save_results(results, output_file)
        
        print(f"\n✅ Validation complete! Check {output_file} for detailed results.")
        
        # Quick summary
        print("\n📋 Quick Summary:")
        for model_name, model_results in results['model_tests'].items():
            if 'error' not in model_results:
                identity_test = model_results.get('identity_comparison', {})
                using_graph = identity_test.get('is_using_graph', False)
                print(f"  {model_name}: {'✅ Using graph' if using_graph else '❌ Not using graph'}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Make sure you have:")
        print("  1. Proper data files in the data/ directory")
        print("  2. Correct graph files in data/graphs/")
        print("  3. All required dependencies installed")
        print("\n🔧 To fix common issues:")
        print("  - Check that your EpiDataLoader parameters match your data")
        print("  - Verify graph files exist in data/graphs/")
        print("  - Make sure all model classes are properly imported")

if __name__ == "__main__":
    main()
