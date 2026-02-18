class WissdatenMountingError(Exception):
    def __init__(self, path: str):
        super().__init__(f"Wissdaten not mounted at path: {path}")