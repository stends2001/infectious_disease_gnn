from typing import Literal

Country         = Literal['germany','hungary']
Level           = Literal['nuts1', 'nuts2', 'nuts3']

ColumnType      = Literal['context','feature','target','pred','split']

DataSetSplit    = Literal['train','val','test']