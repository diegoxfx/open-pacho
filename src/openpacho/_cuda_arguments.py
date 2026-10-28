"""Host-only CUDA launch parameters; no optional runtime imports.

The driver copies argument values during cuLaunchKernel. Device allocations
remain retained separately by the backend until the stream completes.
"""

import ctypes

import numpy as np

_SCALARS = {
    np.bool_: ctypes.c_bool,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
}


class KernelArguments:
    """One bounded parameter vector per compiled kernel, rebound under its lock.

    Stores numeric pointer values, not allocations or caller-owned scalar arrays.
    CUDA device addresses are 64-bit even where a host pointer has another size.
    """

    def __init__(self, args, allocation_type):
        self.types = tuple(type(arg) for arg in args)
        self.is_pointer = tuple(kind is allocation_type for kind in self.types)
        try:
            self.values = tuple(
                (ctypes.c_uint64 if pointer else _SCALARS[kind])()
                for kind, pointer in zip(self.types, self.is_pointer)
            )
        except KeyError as exc:
            raise TypeError(
                "CUDA parameters require allocations or supported NumPy scalars"
            ) from exc
        self.pointers = (ctypes.c_void_p * len(args))(
            *(ctypes.addressof(value) for value in self.values)
        )
        self.address = ctypes.addressof(self.pointers)

    def bind(self, args):
        if len(args) != len(self.values):
            raise TypeError("CUDA kernel parameter count changed")
        for arg, kind, pointer, value in zip(args, self.types, self.is_pointer, self.values):
            if type(arg) is not kind:
                raise TypeError("CUDA kernel parameter type changed")
            value.value = arg.ptr if pointer else arg
        return self.address
