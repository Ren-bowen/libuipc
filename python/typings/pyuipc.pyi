import numpy
from typing import overload

class SmartObjectA:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def view(self) -> numpy.ndarray[numpy.float64]: ...

class SmartObjectB:
    def __init__(self) -> None: ...
    def view(self) -> numpy.ndarray[numpy.float64]: ...

def create_smart_object() -> SmartObjectA: ...
def receive_smart_object(arg0: SmartObjectA) -> None: ...
@overload
def view(arg0: SmartObjectA) -> numpy.ndarray[numpy.float64]: ...
@overload
def view(arg0: SmartObjectB) -> numpy.ndarray[numpy.float64]: ...
