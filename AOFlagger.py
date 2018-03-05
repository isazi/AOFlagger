import argparse
import numpy
import kernel_tuner

import Statistics


def tune_statistics_1D(input_size):
    # First kernel
    kernel = Statistics.Statistics1D(input_size)
    tuning_parameters = dict()
    tuning_parameters["type"] = ["float"]
    tuning_parameters["block_size_x"] = [2**x for x in range(5, 11)]
    tuning_parameters["items_per_thread"] = [2**x for x in range(0, 8)]
    tuning_parameters["thread_blocks"] = [2**x for x in range(0, 17)]
    constraints = ["(thread_blocks * block_size_x * items_per_thread) <= " + str(input_size),
                   "(" + str(input_size) + "- math.ceil(" + str(input_size)
                   + " / thread_blocks) * (thread_blocks - 1)) >= int(block_size_x / 2)"]
    # Data
    data = numpy.random.randn(input_size).astype(numpy.float32)
    statistics = numpy.zeros(max(tuning_parameters["thread_blocks"]) * 3).astype(numpy.float32)
    kernel_arguments = [data, statistics]
    # Control data
    control_arguments = [None, numpy.asarray([input_size, data.mean(), data.var()])]
    try:
        results = kernel_tuner.tune_kernel("compute_statistics_1D", kernel.generate_cuda, "thread_blocks",
                                           kernel_arguments, tuning_parameters, lang="CUDA", restrictions=constraints,
                                           grid_div_x=[], iterations=3, answer=control_arguments, verify=kernel.verify,
                                           atol=1.0e-03)
    except Exception as error:
        print(error)
    # Second kernel

if __name__ == "__main__":
    # Parse command line
    parser = argparse.ArgumentParser(description="AOFmcKT: AOFlagger many-core Kernels Tuner")
    parser.add_argument("--tune_statistics_1D", help="Tune \"compute_statistics_1D()\" kernel.",
                        action="store_true")
    parser.add_argument("--input_size", help="Input size.", required=True, type=int)
    parser.add_argument("--language", help="Language: CUDA or OpenCL.", choices=["CUDA", "OpenCL"], required=True)
    arguments = parser.parse_args()
    if arguments.tune_statistics_1D is True:
        tune_statistics_1D(arguments.input_size)
