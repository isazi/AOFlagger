import kernel_tuner
import numpy


#
# Test swap()
#
def swap_control(data, x, y):
    temp = data[x]
    data[x] = data[y]
    data[y] = temp


def test_swap(data_control, data_device):
    numpy.isclose(data_control, data_device, atol=1.0e-06)


def swap():
    with open("swap.cu") as file:
        kernel_source = file.read()
    input_size = 8192
    data_device = numpy.random.randn(input_size).astype(numpy.float32)
    data_control = data_device
    swap_control(data_control, 0, input_size - 1)
    arguments_device = [data_device, numpy.uint32(0), numpy.uint32(input_size - 1)]
    arguments_control = [data_control, None, None]
    tuning_parameters = {"block_size_x" : {1024}}
    kernel_tuner.tune_kernel("swap", kernel_source, input_size, arguments_device, tuning_parameters,
                             answer=arguments_control, verify=test_swap, quiet=True)


#
# Run the tests
#
if __name__ == "__main__":
    print("Testing function: swap()")
    swap()
