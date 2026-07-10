class InvalidGraphStructure(Exception):

    def __init__(self, msg: str):
        super().__init__(msg)

class InvalidGraphObject(Exception):

    def __init__(self, msg: str):
        super().__init__(msg)

class MissingColumnError(Exception):
    def __init__(self, col: str, df_name: str):
        msg = f'column {col} not found in {df_name}'
        super().__init__(msg)