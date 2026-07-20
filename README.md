# GNN - based epidemiological - predictions
Systematic evaluation of graph structure choice in Graph Neural Network-based 
subnational epidemic forecasting. Companion codebase for [paper title].

## Project Structure

```
data/
notebooks/
src/
├── dataloading
│   ├── columnregistration
│   ├── dataloaders
│   │   ├── baselinedataloader
│   │   └── deepdataloader
│   ├── epiconfig
│   └── epidataorchestration
├── evaluation
├── experimenthandling
├── graphconstruction
├── models
│   ├── base
│   │   ├── basemodel
│   │   └── predictions
│   ├── baseline
│   │   ├── baselinemode.py
│   │   ├── climatology.py
│   │   └── persistence.py
│   ├── deep
│   │   ├── architectures
│   │   ├── deepmodel
│   │   └── strategies
│   └── utils.py
└── utils
    ├── colors.py
    ├── constants.py
    ├── exceptions.py
    ├── helpers.py
    ├── pathmanager.py
    ├── textformatting.py
    └── types.py
```

## Workflow
```mermaid
flowchart TD
    epicfg@{shape: rounded, label: "epiconfig"}
    data@{shape: docs, label: "data"}
    epido@{shape: rounded, label: "epidataorchestrator"}    
    
    bldl@{label: "baselinedataloader"}
    dddl@{label: "deepdataloader"}
    grph@{label: "graph"}

    blml@{shape: hex, label: "baselinemodel (persistence/climatology)"}
    dml@{shape: hex, label: "deepmodel (GCN/GAT)"}    

    preds@{shape: rect, label: "forecasting"}
    eval@{shape: rect, label: "evaluation"}
    
    subgraph Experiment
    subgraph Data Preparation
    epicfg  --> epido
    data    --> epido
    epido   --> bldl
    epido   --> dddl
    grph    --> dddl
    end

    subgraph Models
    bldl    --> blml
    dddl    --> dml
    end

    subgraph Model Evaluation
    blml    --> preds
    dml     --> preds
    preds   --> eval
    end
    end
```