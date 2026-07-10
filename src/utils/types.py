from typing import Literal
Country = Literal['germany', 'hungary']
Level   = Literal['nuts1', 'nuts2', 'nuts3']
GraphType=Literal['identity','geographical_contiguity','gravity_model','random','fully_connected']
GraphNormType=Literal['minmax','symmetric','rowwise']