from openpacho.opencl_backend.opencl_array import OpenCLArray

def main():
    data = [1.0, 2.0, 3.0, 4.0]
    gpu_arr = OpenCLArray(host_data=data)
    print("Initially on GPU:", gpu_arr.to_host())

    # Update GPU data
    gpu_arr.from_host([5.0, 6.0, 7.0, 8.0])
    print("Updated on GPU:", gpu_arr.to_host())

if __name__ == "__main__":
    main()
