
import kernel_tuner
import numpy


# Test swap()
def test_swap():
    with open("swap.cu") as file:
        kernel_source = file.read()
    input_size = 8192
    array_device = numpy.random.randn(input_size).astype(numpy.float32)
    array_host = array_device
    arguments = [array_device, numpy.uint32(0), numpy.uint32(8191)]
    parameters = {}
    kernel_tuner.run_kernel("swap", kernel_source, input_size, arguments, parameters, quiet=True)
    if abs(array_host[0] - array_device[8191]) > 1.0e-06:
        print("Error: " + str(array_host[0]) + " != " + str(array_device[8191]))
        return -1


# Run the tests
if __name__ == "__main__":
    print("Testing function: swap()")
    test_swap()
