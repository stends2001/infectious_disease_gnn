from typing import Literal, List, Callable
from pathlib import Path
import inspect
from abc import ABC

from ...utils.pathmanager import PathManager
from ...utils.types import Country, Level

def registered_property(func: Callable) -> Callable:
    """
    Marks a property and ensures the returned path exists.
    Must be used BEFORE @property.
    """
    def wrapper(self):
        path = func(self)

        if not isinstance(path, Path):
            raise TypeError(f"{func.__name__} did not return a Path")

        return path

    setattr(wrapper, '_is_path', True)
    return wrapper

def get_registered_properties(cls: type) -> List[str]:
    """
    Returns names of all properties decorated with @registered_property.
    """
    return [
        name
        for name, attr in inspect.getmembers(cls)
        if isinstance(attr, property) and getattr(attr.fget, '_is_path', False)
    ]

class EpiPathsManager(ABC):
    """
    Parent class of EpiPathsManagerGermany and EpiPathsManagerNetherlands.
    This is a  Utility class of EpiConfig that deals with the paths.
    Get paths using `.get()` method: useful exception raised when path cannot be found.

    NOTE
    ----
    Netherlands and Germany have different types of data available, so many properties are
    not shared. The once that are, however, are set in this parent class.
    """
    def __init__(self, 
                 country: Country,
                 level:   Level):
        
        self.properties = get_registered_properties(self.__class__)
        self.country    = country
        self.level      = level
        self.pm         = PathManager()

    def get(self, property: str) -> Path:
        if property not in self.properties:
            raise ValueError(f'{property} is not a known property of {self.__class__.__name__}. Valid Properties are {self.properties}')
        
        return getattr(self, property)

    # ======= PATHS SHARED AMONG PATHMANAGERS ====== #        
    @property
    @registered_property    
    def data_env(self) -> Path:
        return self.pm.data / 'data'    

    @property
    @registered_property    
    def population_size(self) -> Path:
        """Path to population size CSV file."""
        return self.data_env / f'{self.country}/sociodemography/population_size.csv'  

    @property
    @registered_property   
    def region_harmonization(self) -> Path:
        """Path to NUTS names file."""
        return self.data_env / f'{self.country}/geospatial/level_harmonization.tsv'

    @property
    @registered_property   
    def shapefile(self) -> Path:    
        """Path to shapefile of the country (including all levels)"""
        return self.data_env / f'{self.country}/geospatial/level_shapes.shp'        
    
    @property
    @registered_property   
    def tokenization_map(self) -> Path:    
        """Path to shapefile of the country (including all levels)"""
        return self.data_env / f'{self.country}/graphs/{self.level}/tokenization_map.json'       
    
    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}>"
        return representation    

class EpiPathsManagerGermany(EpiPathsManager):
    """ 
    EpiPathsManger for Germany specifically
    """

    def __init__(self,
                 disease: str,
                 level: Level):
        
        self.disease = disease

        # set registry of paths in parent class
        super().__init__(country = 'germany', level = level)

    @property
    @registered_property   
    def cases(self) -> Path:
        """Path to disease CSV file."""
        return self.data_env / 'germany/epidemiology' / f'{self.disease}.csv'
    
    @property
    @registered_property   
    def gisd(self) -> Path:
        """Path to gisd CSV file."""
        return self.data_env / f'germany/sociodemography/gisd.csv'    

    @property
    @registered_property   
    def population_age(self) -> Path:
        """Path to population age CSV file."""
        return self.data_env / f'germany/sociodemography/population_age.csv'        
    
    @property
    @registered_property      
    def kreise_classes(self) -> Path:
        """Path to kreise classification CSV file."""
        return self.data_env / f'germany/sociodemography/kreis_classes.csv' 
    
    @property
    @registered_property   
    def border_regions(self) -> Path:
        """Path to borders CSV file."""
        return self.data_env / f'germany/sociodemography/border_regions.csv' 
    
    @property
    @registered_property     
    def population_density(self) -> Path:
        """Path to population density CSV file."""
        return self.data_env / f'{self.country}/sociodemography/population_density.csv'         

class EpiPathsManagerNetherlands(EpiPathsManager):
    """ 
    EpiPathsManger for Netherlands specifically
    """

    def __init__(self,
                 disease: str,
                 level: Level):
        
        self.disease = disease

        # set registry of paths in parent class
        super().__init__(country = 'netherlands', level = level)

    @property
    @registered_property   
    def cases(self) -> Path:
        """Path to disease CSV file."""
        return self.data_env / f'netherlands/epidemiology/{self.disease}.csv'
    
    @property
    @registered_property     
    def population_density(self) -> Path:
        """Path to population density CSV file."""
        return self.data_env / f'{self.country}/sociodemography/population_density.csv'         

class EpiPathsManagerHungary(EpiPathsManager):
    """ 
    EpiPathsManger for Hungary specifically
    """

    def __init__(self,
                 disease: str,
                 level: Level):
        
        self.disease = disease

        # set registry of paths in parent class
        super().__init__(country = 'hungary', level = level)

    @property
    @registered_property   
    def cases(self) -> Path:
        """Path to disease CSV file."""
        return self.data_env / f'hungary/epidemiology/{self.disease}.csv'
                