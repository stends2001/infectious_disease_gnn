# Influenza - weekly incidence predictions

The main experiment I have been running is the following:

```
epiconfig = EpiConfig(
        disease             = 'influenza',

        horizon_leadtime    = 3,
        sequence_length     = 4,

        feature_popsize     = True,      
        feature_popdens     = True,
        
        log_transform       = ['incidence']     
        )   
```

Please find more information on the epiconfig and how it translates into an actual dataset in src.dataloading.epiconfig and src.dataloading.epidataorchestration.


I am now comparing the following graph structures:
- graph1: identity graph
- graph2: 