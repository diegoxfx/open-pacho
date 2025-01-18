from typing import Dict
from openpacho.core.backend import OpenPachoBackend

# A global registry for backend types
_backend_registry: Dict[str, OpenPachoBackend] = {}

def register_backend(name: str, backend_cls: OpenPachoBackend):
    if name in _backend_registry:
        return
    _backend_registry[name] = backend_cls

def get_backend(name: str) -> OpenPachoBackend:
    backend_cls = _backend_registry.get(name)
    if not backend_cls:
        raise ValueError(f"Backend '{name}' is not registered.")
    return backend_cls()
