from ...utils import blackcolor, paired_colors

MODELSREGISTRY = {
        'unknown'               : 0,            # all unknowns will be shown in black
        'PersistenceModel'      : 1,
        'NaiveLinearModel'      : 2,
        'SpatioTemporalXGBModel': 3,
        'GATv2Model'            : 4,
        'TGCNModel'             : 5,
        'GConvLSTMModel'        : 6
    }    

MODELSCOLORPALETTE = [blackcolor] + paired_colors