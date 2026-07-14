
# ======= EPICONFIG ======= #
class EpiConfigValidationError(Exception):
    """
    errors are to be raised!
    """    
    def __init__(self, msg: str):
        super().__init__(msg)

class EpiConfigLimitationError(Exception):
    """
    """    
    def __init__(self, msg: str):
        super().__init__(msg)  

class InvalidCovariatePath(Exception):
    """
    """    
    def __init__(self, msg: str):
        super().__init__(msg)      

class IncompatibleEpiConfigs(Exception):

    def __init__(self, msg: str):
        super().__init__(msg)