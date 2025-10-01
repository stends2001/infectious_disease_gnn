# Spatiotemporal modelling of infectious diseases

## Abstract
This project aims to develop and validate machine learning algorithms using data on vaccine-preventable diseases to study and forecast their spatio-temporal dynamics. Given its public health impact, measles is the primary focus. Unlike many other diseases, measles outbreaks often start with cases imported from abroad.

By integrating socio-demographic, epidemiological, and travel data, this project aims to improve understanding of transmission patterns and build predictive models for future outbreaks, especially for measles. Early work has tested various temporal models, using influenza as a proof of concept. Influenza is seasonal and well-documented in Germany, and simple graph neural networks (GNNs)—deep learning models that capture spatial relationships—show promising predictive power on this dataset.

The next step is to develop GNN models suited for datasets with many instances of no reported cases, typical for measles. By analysing how outbreaks spread spatially, the project will assess how well these models predict future cases. The goal is to advance both modelling methods and knowledge of vaccine-preventable diseases.

## Current update
I have an environment up and running. I am running mostly experiments on influenza casenumbers in Germany on Bundeslander (NUTS1), NUTS2, and Kreis (NUTS3) level. My goal in this is to create an environment in which I test what works and what doesn't work. I've been trying out some experiments with a bunch of different graphs:
- identity graph => each node is only connected to itself
- boolean neighbors => each node is only connected to those directly surrounding it
- gravity models - variations

I've also started with some baseline spatial - GCN vs spatio-temporal GNNs. For now, nothing meaningful to report.

One thing I struggle with is that I don't how to weight my predictions. Should I forecast 1 timestep ahead, or use a horizon of a couple timesteps. I've been doing the latter one, but then determining the loss over all of them rather than on the first or last timestep. I further find that the 1 - step ahead prediction is basically a repetition of the most-recently observed trend.

## Project goals:
- different graph - systems 
	- pertussis => children?
	- influenza => neighborhood?
	- measles => travel?
- develop a spatiotemporal - pipeline whatever tha may be
- for measles specifically, relate local German epidemiology to international epidemiology
    - have data per airport in Germany: another gravity model layer on top to model the spread of incoming flights, and associated risk depending on 
      origin of flight
- childhood diseases => school holidays
- scenario development => long term predictions
      
## To Dos
- Denormalization
- Loss class
- Metrics
- Experiment - class and -runner
- Dashboard -> interact with models
- Config and results to be optimized
- Go through Notebooks

## Literature Review

[[Kraemer, 2024]]

[[Croft, 2023]]

[[Jeong, 2025]]

[[Wang, 2025]] use a SIR based model to predict parameters relevant to the spread over multiple regions.

[[Liu, 2024]] propose the GRGNN, the GRU-based GNN. 

[[Grasslyl, 2006]] --> seasonality

## Resources
https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/recurrent/a3tgcn_example.py

https://github.com/twitter-research/tgn