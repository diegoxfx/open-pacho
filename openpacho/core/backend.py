from __future__ import annotations
import typing
from abc import abstractmethod
from openpacho.core.utils import SingletonABC


if typing.TYPE_CHECKING:
    from openpacho.core.array import OpenPachoArray

class OpenPachoBackend(SingletonABC):
    def __init__(self, backend_name):
        self.backend_name = backend_name

    @abstractmethod
    def vector_dot(self, x: OpenPachoArray, y: OpenPachoArray):
        pass

    @abstractmethod
    def matrix_vec(self, x: OpenPachoArray, y: OpenPachoArray):
        pass