from typing import TYPE_CHECKING, List, assert_never
import pandas as pd

from .exceptions import InvalidCovariatePath, EpiConfigLimitationError, EpiConfigValidationError, EpiConfigWarning
from ...issues.issuereport import IssueReport, IssueBase

if TYPE_CHECKING:
    from .epiconfig import EpiConfig

class EpiConfigValidator:
    """
    This is a  Utility class of EpiConfig that deals with the validation of input.
    Simply call `.validate()`. Warngings that do not thro exceptions may be printed.
    Alternatively, an IssueReported may be returned.
    """

    def __init__(self,
                 epiconfig: 'EpiConfig'):
        
        self.epiconfig = epiconfig 

    def validate(self):
        exceptions: List[IssueBase] = []
        exceptions = self._datapaths(exceptions)
        exceptions = self._current_limitations(exceptions)      
        exceptions = self._input(exceptions)  
        
        self._warnings()

        if len(exceptions) > 0:
            raise IssueReport(exceptions, context = "EpiConfig could not be created")                    

    def _datapaths(self, exceptions: List[IssueBase]):

        for property in self.epiconfig.path_manager.properties:
            path_attr = self.epiconfig.path_manager.get(property)

            if not path_attr.exists():
                exceptions.append(InvalidCovariatePath(f"path {property} not found: {path_attr}"))    

        return exceptions        
    
    def _current_limitations(self, exceptions: List[IssueBase]) -> List[IssueBase]:
        """
        Validates any issues in the initialization of an EpiConfig instance. 
        These represent CURRENT limitations, which are also things for me to develop further.
        An CurrentEpiConfigError is thrown suggesting to adjust the input.
        """

        # temporal frequency
        if self.epiconfig.temporal_frequency not in ['m','w','d']:
            exceptions.append(EpiConfigLimitationError(f'invalid valid for temporal_frequency (currently). Value must be in ["m","w","d"]'))

        # predicting deltas
        if self.epiconfig.predict_difference:
            exceptions.append(EpiConfigLimitationError(
                f"predict_difference is currently not supported."
            ))               

        if self.epiconfig.predict_difference and self.epiconfig.horizon_leadtime > 1:
            exceptions.append(EpiConfigLimitationError(
                f"predict_difference=True is only supported with horizon_leadtime=1 (currently) "
                f"Got horizon_leadtime={self.epiconfig.horizon_leadtime}. "
                f"For multi-step forecasting with deltas, set horizon_leadtime=1 and use horizon_size > 1 instead."
            ))            
        
        # features
        # gisd
        if self.epiconfig.feature_gisd:
            if pd.to_datetime(self.epiconfig.max_date) > pd.to_datetime('2021-12-31'):
                exceptions.append(EpiConfigLimitationError('Currently GISD data only available until 2021 while simulation max date exceeds that. Either remove the gisd data as feature, or decrease the timespawn.'))
            if self.epiconfig.level == 'nuts1':
                exceptions.append(EpiConfigLimitationError('Currently GISD data only available for nuts levels 2 and 3. Please Adjust'))                
        
        if self.epiconfig.feature_climateology:
            exceptions.append(EpiConfigLimitationError('Currently no climateology features supported'))      

        if self.epiconfig.country == 'netherlands' and self.epiconfig.disease != 'covid_daily':
            exceptions.append(EpiConfigLimitationError('Currently only covid_daily available for the Netherlands'))   

        return exceptions
    
    def _input(self, exceptions: List[IssueBase]) -> List[IssueBase]:
        """
        Validates discrepancies in the initialization of an EpiConfig instance. These represent
        actual issues or errors, so an EpiConfigError is thrown suggesting to adjust the input.
        """
        # temporal-related 
        if self.epiconfig.horizon_size < 1:
            exceptions.append(EpiConfigValidationError(f"horizon_size must be >= 1, got {self.epiconfig.horizon_size}"))
        
        if self.epiconfig.horizon_leadtime < 1:
            exceptions.append(EpiConfigValidationError(f"horizon_leadtime must be >= 1, got {self.epiconfig.horizon_leadtime}"))
        
        if self.epiconfig.sequence_length < 1:
            exceptions.append(EpiConfigValidationError(f"sequence_length must be >= 1, got {self.epiconfig.sequence_length}"))
        
        if self.epiconfig.lag_num < 1:
            exceptions.append(EpiConfigValidationError(f"number of lags must be >= 1, got {self.epiconfig.lag_num}"))
        
        if self.epiconfig.time_index_d and self.epiconfig.disease != 'covid_daily':
            exceptions.append(EpiConfigValidationError(f'time_index_d is only relevant to disease covid_daily'))

        # task-related
        if self.epiconfig.target_column == 'incidence' and self.epiconfig.prediction_mode == 'classification':
            exceptions.append(EpiConfigValidationError(f'Invalid combination of target == "incidence" prediction_mode as "classification"'))
        
        if self.epiconfig.quantiles:
            if not isinstance(self.epiconfig.quantiles, list):
               exceptions.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.epiconfig.quantiles}). Must be a List[float]'))
            
            middle_idx = int(len(self.epiconfig.quantiles) / 2)

            for quantile in self.epiconfig.quantiles:
                if quantile >= 1 or quantile <= 0:
                   exceptions.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.epiconfig.quantiles}). Must be a List of values 0 < quantile < 1'))

            if not len(self.epiconfig.quantiles) % 2:
                exceptions.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.epiconfig.quantiles}). Must be of odd length'))
            
            if self.epiconfig.quantiles[int(len(self.epiconfig.quantiles) / 2)] != 0.5:
                exceptions.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.epiconfig.quantiles}). Must be symmetric around quantile 0.5'))       

            if len(self.epiconfig.quantiles) > 1:

                for idx_l in range(0,middle_idx):
                    idx_r = len(self.epiconfig.quantiles) - idx_l - 1

                    if self.epiconfig.quantiles[idx_l] + self.epiconfig.quantiles[idx_r] != 1.0:                 
                        exceptions.append(EpiConfigValidationError(f'Invalid input for quantiles ({self.epiconfig.quantiles}). Must be symmetric around quantile 0.5'))                                   

        # country-related
        match (self.epiconfig.country, self.epiconfig.level):

            case ('netherlands', 'nuts3'):
                exceptions.append(EpiConfigValidationError(f'Invalid input for (country, level). nuts3 is unavailable for the Netherlands'))            

            case ('netherlands', 'nuts1' | 'nuts2' | 'ggd' | 'lau'):
                pass 

            case ('germany', 'ggd' | 'lau'):
                exceptions.append(EpiConfigValidationError(f'Invalid input for (country, level). {self.epiconfig.level} is unavailable for Germany'))            

            case ('hungary', 'ggd' | 'lau'):
                exceptions.append(EpiConfigValidationError(f'Invalid input for (country, level). {self.epiconfig.level} is unavailable for Hungary'))  

            case ('germany', 'nuts1' | 'nuts2' | 'nuts3'):
                pass

            case ('hungary', 'nuts1' | 'nuts2' | 'nuts3'):
                pass            

            case _:
                assert_never(self.epiconfig.country, self.epiconfig.level)                

        if self.epiconfig.country in ['netherlands','hungary']:

            unsupported_features = [
                ("feature_borders", self.epiconfig.feature_borders),
                ("feature_gisd", self.epiconfig.feature_gisd),
                ("feature_kreise_classes", self.epiconfig.feature_kreise_classes),
                ("feature_popage", self.epiconfig.feature_popage),
            ]

            if self.epiconfig.country == 'hungary':
                unsupported_features.append( ("feature_popdens", self.epiconfig.feature_popdens))

            invalid_features = [name for name, val in unsupported_features if val]

            for feature in invalid_features:
                exceptions.append(EpiConfigValidationError(f'Invalid feature. {feature} is unavailable for {self.epiconfig.country}.'))                  

        return exceptions
        
    def _warnings(self):
        """
        Validates some combinations of inputs that are likely not meant as such, and shouldn't disrupt the pipeline any further. 
        A EpiConfigWarning is thrown, not an exception
        """
        if self.epiconfig.prediction_mode != 'regression' and self.epiconfig.incidence_scalar != 10_000:
            w = EpiConfigWarning('incidence_scalar will not be taken into account when using prediction_mode != "regression"')
            print(w)

        if self.epiconfig.quantiles is not None and self.epiconfig.prediction_mode != 'regression':
            w = EpiConfigWarning('quantiles will not be taken into account when using prediction_mode != "regression"')                
            print(w)
            
        match (self.epiconfig.country, self.epiconfig.level):

            case ('netherlands', 'nuts1'):
                w = EpiConfigWarning('Netherlands - nuts1 units are very large (n = 4). Predictions may not be particulary informative.')
                print(w)

            case ('netherlands', 'nuts2'):
                w = EpiConfigWarning('Netherlands - nuts2 units are very large (n = 12). Predictions may not be particulary informative.')
                print(w)

            case ('hungary', 'nuts1'):
                w = EpiConfigWarning('Hungary - nuts1 units are very large (n = ?). Predictions may not be particulary informative.')
                print(w)

            case ('hungary', 'nuts2'):
                w = EpiConfigWarning('Hungary - nuts2 units are very large (n = ?). Predictions may not be particulary informative.')
                print(w)                

            case ('germany', 'nuts1'):
                w = EpiConfigWarning('Germany - nuts1 units are very large (n = 16). Predictions may not be particulary informative.')
                print(w)    
    
    def __repr__(self) -> str:
        representation = f"<{self.__class__.__name__}>"
        return representation 