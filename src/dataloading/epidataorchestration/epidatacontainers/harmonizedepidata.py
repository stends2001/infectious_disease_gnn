import pandas as pd 
from dataclasses import dataclass
from typing import Optional

from ..utils.issues import NonExistentAttributeEpiDataContainer
from ....utils.textformatting import checkmark

@dataclass
class HarmonizedEpiData:
    """
    """        
    epidata:            pd.DataFrame

    _population_size:    Optional[pd.DataFrame] = None
    _population_density: Optional[pd.DataFrame] = None 
    _population_age:     Optional[pd.DataFrame] = None    
    _gisd:               Optional[pd.DataFrame] = None  
    _kreise_classes:     Optional[pd.DataFrame] = None
    _borders:            Optional[pd.DataFrame] = None
    _vacmap:             Optional[pd.DataFrame] = None

    @property
    def population_size(self) -> pd.DataFrame:
        df = self._population_size
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_size')
        return df 

    @property
    def population_density(self) -> pd.DataFrame:
        df = self._population_density
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_density')
        return df         

    @property
    def population_age(self) -> pd.DataFrame:
        df = self._population_age
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'population_age')
        return df             

    @property
    def gisd(self) -> pd.DataFrame:
        df = self._gisd
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'gisd')
        return df         

    @property
    def kreise_classes(self) -> pd.DataFrame:
        df = self._kreise_classes
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'kreise_classes')
        return df     

    @property
    def borders(self) -> pd.DataFrame:
        df = self._borders
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'borders')
        return df   

    @property
    def vacmap(self) -> pd.DataFrame:
        df = self._vacmap
        if df is None: 
            raise NonExistentAttributeEpiDataContainer(self.__class__.__name__, 'vacmap')
        return df                        

    def __repr__(self):
        representation = f"<{self.__class__.__name__}(epidata {checkmark}"

        if self._population_size is not None:
            representation += f", population_size {checkmark}"

        if self._population_density is not None:
            representation += f", population_density {checkmark}"
        
        if self._population_age is not None:
            representation += f", population_age {checkmark}"

        if self._gisd is not None:
            representation += f", gisd {checkmark}"

        if self._kreise_classes is not None:
            representation += f", kreise_classes {checkmark}"

        if self._borders is not None:
            representation += f", borders {checkmark}"            

        if self._vacmap is not None:
            representation += f", vacmap {checkmark}"    

        representation += ")>"
        return representation
