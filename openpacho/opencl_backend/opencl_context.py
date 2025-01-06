import pyopencl as cl

class OpenCLContextManager:
    _instance = None

    def __new__(cls, device_type="GPU"):
        """
        Implements the singleton pattern so that we only have
        one instance of the context manager throughout the app.
        """
        if cls._instance is None:
            cls._instance = super(OpenCLContextManager, cls).__new__(cls)
            cls._instance._init_context(device_type)
        return cls._instance

    def _init_context(self, device_type):
        # Get platforms
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
                selected_device = devices[0]
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
