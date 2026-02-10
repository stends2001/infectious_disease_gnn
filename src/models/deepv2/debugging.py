from dataclasses import dataclass
from typing import Any, List 

from ...utils.textformatting import checkmark

class DeepModelDebuggingError(Exception):
    def __init__(self, message: str):
        super().__init__(f'Error in Model debugging {message}')


@dataclass
class DebuggingLine:
    got: Any 
    expected: Any

@dataclass
class ModelDebuggingReport:

    lines:      List[DebuggingLine]

    def __iter__(self):
        return iter(self.lines)
    
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx: int) -> DebuggingLine:
        return self.lines[idx]

    def validate(self):
        for ll in self.lines:
            if ll.got != ll.expected:
                raise DeepModelDebuggingError(f'got {ll.got} but expected {ll.expected}')
        print(f'{checkmark} Debugging went fine')
      