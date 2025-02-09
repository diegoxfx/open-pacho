from openpacho.core.array import OpenPachoArray

def transpose(A: OpenPachoArray):
    return A.backend.transpose(A)