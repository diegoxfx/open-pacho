# Global or module-level variable
_device_type = "GPU"
_device_index = None

def set_device_type(device_type, device_index=None):
    global _device_type
    global _device_index
    _device_type = device_type.upper()
    _device_index = device_index

def get_device_type():
    return _device_type

def get_device_index():
    return _device_index