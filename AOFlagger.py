import kernel_tuner
import numpy

import Statistics


def tune_statistics():
    input_size = 25000
    input_chunks = 50

    data = numpy.random.randn(input_size).astype(numpy.float32)
    statistics = numpy.zeros(input_size)
    kernel_arguments = [data, statistics]

    tuning_parameters = dict()
    tuning_parameters["type"] = "float"
    tuning_parameters["threads_per_block"] = [threads for threads in range(32, 1024 + 1, 32)]
    tuning_parameters["threads_per_block_other_dim"] = [1]
    tuning_parameters["items_per_thread"] = [items for items in range(1, 256, 1)]
    tuning_parameters["items_per_block"] = [input_size / input_chunks]
    constraints = ["(" + str(input_size) + " % items_per_block) == 0",
                   "(items_per_block % (threads_per_block * items_per_thread)) == 0"]
    block_size_names = ["threads_per_block", "threads_per_block_other_dim", "threads_per_block_other_dim"]
    try:
        results = kernel_tuner.tune_kernel("compute_statistics_1D", Statistics.Statistics.generate_cuda, input_size,
                                           kernel_arguments, tuning_parameters, lang="CUDA", restrictions=constraints,
                                           block_size_names=block_size_names, iterations=3)
    except Exception as error:
        print(error)


if __name__ == "__main__":
    tune_statistics()