import kernel_tuner
import numpy

import Statistics


def tune_statistics():
    input_size = 25000

    # Kernel
    kernel = Statistics.Statistics1D(input_size)
    tuning_parameters = dict()
    tuning_parameters["type"] = ["float"]
    tuning_parameters["block_size_x"] = [2**x for x in range(5, 11)]
    tuning_parameters["items_per_thread"] = [2**x for x in range(0, 8)]
    tuning_parameters["thread_blocks"] = [2**x for x in range(0, 17)]
    constraints = ["(thread_blocks * block_size_x * items_per_thread) <= " + str(input_size)]
    # Data
    data = numpy.random.randn(input_size).astype(numpy.float32)
    statistics = numpy.zeros(max(tuning_parameters["thread_blocks"]) * 3).astype(numpy.float32)
    kernel_arguments = [data, statistics]
    # Control data
    control_arguments = [None, numpy.asarray([input_size, data.mean(), data.var()])]

    try:
        results = kernel_tuner.tune_kernel("compute_statistics_1D", kernel.generate_cuda, "thread_blocks",
                                           kernel_arguments, tuning_parameters, lang="CUDA", restrictions=constraints,
                                           grid_div_x=[], iterations=3, answer=control_arguments,
                                           verify=Statistics.Statistics1D.verify, atol=1.0e-03)
    except Exception as error:
        print(error)


if __name__ == "__main__":
    tune_statistics()