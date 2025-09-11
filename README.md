# Spatio-temporal modelling of infectious diseases in Germany

## Abstract
Infectious diseases remain a significant threat to public health worldwide, with their spread often exhibiting complex spatial and temporal patterns influenced by various regional factors. Recently, Graph Neural Networks (GNNs) have emerged as a powerful tool for modeling structured data, making them well-suited for capturing the intricate relationships inherent in epidemiological data across interconnected regions.

This project aims to develop a comprehensive pipeline for applying GNN-based models to regional infectious disease prediction across multiple diseases in Germany, using data extracted from SurvStat. The pipeline will streamline data processing, model training, evaluation, and interpretation, providing a flexible framework adaptable to different diseases and regional settings. By leveraging GNNs’ ability to integrate spatial connectivity and temporal dynamics, the project seeks to improve forecasting accuracy and support data-driven public health decision-making.

Ultimately, this pipeline will facilitate systematic experimentation with various graph architectures and epidemiological datasets, advancing the understanding of how spatial dependencies and regional interactions drive disease spread.

## Methods

## To Dos
- Error in plotting / normalization. Loss hasn't changed, but plots have!
- Loss class
- Integration of final metrics
- setup benchmarking neural network
- extend GCN layers
- organize project notebooks
- Work with prediction horizon for multiple models

## Literature Review

[[Kraemer, 2024]]

[[Croft, 2023]]

[[Jeong, 2025]]

[[Wang, 2025]] use a SIR based model to predict parameters relevant to the spread over multiple regions.

[[Liu, 2024]] propose the GRGNN, the GRU-based GNN. 