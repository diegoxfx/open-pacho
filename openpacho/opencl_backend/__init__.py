from openpacho.core.backend_registry import register_backend
from openpacho.opencl_backend.opencl_backend import OpenCLBackend
register_backend("opencl", OpenCLBackend)
