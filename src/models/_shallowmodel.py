from ._basemodel import BaseModel, EpiDataLoader, pd, Optional


class ShallowModel(BaseModel):

    def __init__(self, dataloader: EpiDataLoader, name: Optional[str] = None, node_label: bool = True):
        self.node_label = node_label
        super().__init__(dataloader, name)  

        self.dataloaders = self.get_dataloaders(node_label=self.node_label)   

    def get_dataloaders(self, node_label: bool = True):
        dataset = self.dataloader.data['final']
        train   = dataset[dataset['train']]
        val     = dataset[dataset['val']]
        test    = dataset[dataset['test']]

        X_train, y_train, c_train = self._split_Xyc(train, node_label)
        X_val,   y_val,   c_val   = self._split_Xyc(val, node_label)
        X_test,  y_test,  c_test  = self._split_Xyc(test, node_label)
        
        dataloaders = {
            "train": {"X": X_train, "y": y_train, "c" : c_train},
            "val"  : {"X": X_val,   "y": y_val,   "c" : c_val} ,           
            "test" : {"X": X_test,  "y": y_test,  "c" : c_test}
        }
        return dataloaders

    def _split_Xyc(self, df: pd.DataFrame, node_label: bool = True):

        # Conditionally include the node_label column (your id_column)
        if node_label:
            X = df[[self.dataloader.id_column] + self.dataloader.feature_columns]
        else:
            # Exclude id_column
            X = df[self.dataloader.feature_columns]

        y = df[self.dataloader.target_column]
        c = df[[self.dataloader.temporal_column, self.dataloader.id_column]]
        
        return X.reset_index(drop = True), y.reset_index(drop = True), c.reset_index(drop = True)     
