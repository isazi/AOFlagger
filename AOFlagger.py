import kernel_tuner
import numpy

import Statistics


def tune_statistics():
    input_size = 25000

    # Kernel
    kernel = Statistics.Statistics(input_size)
    tuning_parameters = dict()
    tuning_parameters["type"] = "float"
    tuning_parameters["threads_per_block"] = [threads for threads in range(32, 1024 + 1, 32)]
    tuning_parameters["threads_per_block_other_dim"] = [1]
    tuning_parameters["items_per_thread"] = [items for items in range(1, 256, 1)]
    tuning_parameters["thread_blocks"] = [2**x for x in range(0, 16)]
    constraints = []
    block_size_names = ["threads_per_block", "threads_per_block_other_dim", "threads_per_block_other_dim"]
    # Data
    data = numpy.random.randn(input_size).astype(numpy.float32)
    statistics = numpy.zeros(max(tuning_parameters["thread_blocks"]) * 3).astype(numpy.float32)
    kernel_arguments = [data, statistics]
    try:
        results = kernel_tuner.tune_kernel("compute_statistics_1D", kernel.generate_cuda, input_size,
                                           kernel_arguments, tuning_parameters, lang="CUDA", restrictions=constraints,
                                           block_size_names=block_size_names, iterations=3)
    except Exception as error:
        print(error)


if __name__ == "__main__":
    tune_statistics()