import kernel_tuner
import numpy

import Statistics


def tune_statistics():
    input_size = 25000

    # Kernel
    kernel = Statistics.Statistics(input_size)
    tuning_parameters = dict()
    tuning_parameters["type"] = ["float"]
    tuning_parameters["block_size_x"] = [threads for threads in range(32, 1024 + 1, 32)]
    tuning_parameters["items_per_thread"] = [2**x for x in range(0, 8)]
    tuning_parameters["thread_blocks"] = [2**x for x in range(0, 17)]
    constraints = ["(thread_blocks * block_size_x * items_per_thread) <= " + str(input_size)]
    # Data
    data = numpy.random.randn(input_size).astype(numpy.float32)
    statistics = numpy.zeros(max(tuning_parameters["thread_blocks"]) * 3).astype(numpy.float32)
    kernel_arguments = [data, statistics]
    # Control data
    control_arguments = [None, numpy.asarray([input_size, data.mean(), data.var()])]

    # Control function
    def verify(control_data, data, atol=1.0e-06):
        counter = 0.0
        mean = 0.0
        variance = 0.0
        for item in range(0, max(tuning_parameters["thread_blocks"]) * 3, 3):
            temp = data[item + 1] - mean
            mean = ((counter * mean) + (data[item] * data[item + 1])) / (counter + data[item])
            variance = variance + data[item + 2] + ((temp * temp) * ((counter * data[item]) / (counter + data[item])))
            counter = counter + data[item]
        return numpy.allclose(control_data, [counter, mean, variance], atol)

    try:
        results = kernel_tuner.tune_kernel("compute_statistics_1D", kernel.generate_cuda, "thread_blocks",
                                           kernel_arguments, tuning_parameters, lang="CUDA", restrictions=constraints,
                                           grid_div_x=[], iterations=3, answer=control_arguments, verify=verify)
    except Exception as error:
        print(error)


if __name__ == "__main__":
    tune_statistics()