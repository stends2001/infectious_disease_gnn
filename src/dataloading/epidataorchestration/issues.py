from ...issues.warnings import Warning

class EpiConfigWarning(Warning):
    def __init__(self, statement: str):
        super().__init__(statement)

class EpiConfigError(Exception):
    def __init__(self, explanation: str):
        statement = "Epiconfig couldn't be loaded" + "\n" + explanation
        super().__init__(statement)

class CurrentEpiConfigError(Exception):
    def __init__(self, explanation: str):
        statement = "Epiconfig couldn't be loaded" + "\n" + explanation
        super().__init__(statement)
