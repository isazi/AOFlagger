
import kernel_tuner
import numpy

FILENAME = "AOFlagger.cu"

with open(FILENAME) as file:
    kernel_source = file.read()


# Test swap()
def test_swap():
    input_size = 8192
    array_device = numpy.random.randn(input_size).astype(numpy.float32)
    # array_host = array_device
    arguments = [array_device, 0, 8191]
    parameters = []
    kernel_tuner.run_kernel("swap", kernel_source, input_size, arguments, parameters)


# Run the tests
if __name__ == "__main__":
    test_swap()
