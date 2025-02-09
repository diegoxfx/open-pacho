import pyopencl as cl
from openpacho.config import get_device_type
from openpacho.config import get_device_index

class OpenCLContextManager:
    _instance = None

    def __new__(cls):
        """
        Implements the singleton pattern so that we only have
        one instance of the context manager throughout the app.
        """
        if cls._instance is None:
            cls._instance = super(OpenCLContextManager, cls).__new__(cls)
            cls._instance._init_context()
        return cls._instance

    def _init_context(self):
        device_type = get_device_type()  # "CPU" or "GPU"
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found.")

        # Attempt to find a device that matches device_type
        selected_device = None
        for platform in platforms:
            devices = []
            if device_type.upper() == "GPU":
                devices = platform.get_devices(device_type=cl.device_type.GPU)
            elif device_type.upper() == "CPU":
                devices = platform.get_devices(device_type=cl.device_type.CPU)

            if devices:
                device_index = get_device_index()
                if device_index is None:
                    print("Multiple OpenCL devices detected:")
                    for i, dev in enumerate(devices):
                        print(f"[{i}] {dev.name} (platform: {dev.platform.name})")
                    device_index = int(input("Select device index: "))
                selected_device = devices[device_index]
                break

        if selected_device is None:
            raise RuntimeError(f"No OpenCL {device_type} device found.")

        self.device = selected_device
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

    def get_context(self):
        return self.context

    def get_queue(self):
        return self.queue

    def get_device(self):
        return self.device

    def get_preferred_work_group_size_multiple(self):
        """
        Query the device for CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE.
        """
        try:
            preferred_multiple = self.device.preferred_work_group_size_multiple
            return preferred_multiple
        except Exception as e:
            print(f"Warning: Could not get preferred work group size multiple. Defaulting to 8. Error: {e}")
            return 8  # Fallback to 8 if query fails