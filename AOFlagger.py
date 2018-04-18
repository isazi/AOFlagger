import argparse
import numpy
import math
import kernel_tuner

import Statistics


def tune_meanandstddev_1D(input_size, language):
    # First kernel
    kernel = Statistics.MeanAndStandardDeviation1D(input_size)
    tuning_parameters_first = dict()
    tuning_parameters_first["type"] = ["float"]
    tuning_parameters_first["block_size_x"] = [2 ** x for x in range(5, 11)]
    tuning_parameters_first["items_per_thread"] = [2 ** x for x in range(0, 8)]
    tuning_parameters_first["thread_blocks"] = [2 ** x for x in range(0, 17)]
    constraints = ["(thread_blocks * block_size_x * items_per_thread) <= " + str(input_size),
                   "(" + str(input_size) + "- math.ceil(" + str(input_size)
                   + " / thread_blocks) * (thread_blocks - 1)) >= int(block_size_x / 2)"]
    data = numpy.random.randn(input_size).astype(numpy.float32)
    triplets = numpy.zeros(max(tuning_parameters_first["thread_blocks"]) * 3).astype(numpy.float32)
    kernel_arguments = [data, triplets]
    control_arguments = [None, numpy.asarray([input_size, data.mean(), data.var()])]
    results_first = dict()
    try:
        if language == "CUDA":
            results_first, platform = kernel_tuner.tune_kernel("compute_statistics_1D_first_step",
                                                               kernel.generate_first_step_cuda, "thread_blocks",
                                                               kernel_arguments, tuning_parameters_first, lang=language,
                                                               restrictions=constraints, grid_div_x=[], iterations=3,
                                                               answer=control_arguments,
                                                               verify=kernel.verify_first_step,
                                                               atol=1.0e-03, quiet=True)
    except Exception as error:
        print(error)
    # Second kernel
    tuning_parameters_second = dict()
    tuning_parameters_second["block_size_x"] = [2 ** x for x in range(1, 11)]
    tuning_parameters_second["thread_blocks"] = [1]
    results_second = dict()
    for blocks in tuning_parameters_first["thread_blocks"]:
        kernel = Statistics.MeanAndStandardDeviation1D(blocks * 3)
        triplets = numpy.random.random_integers(2, 2 + blocks, blocks * 3).astype(numpy.float32)
        statistics = numpy.zeros(2).astype(numpy.float32)
        kernel_arguments = [triplets, statistics]
        constraints = ["block_size_x <= " + str(blocks)]
        control_arguments = [None, numpy.asarray(kernel.generate_control_data_second_step(triplets))]
        try:
            if language == "CUDA":
                results, platform = kernel_tuner.tune_kernel("compute_statistics_1D_second_step",
                                                             kernel.generate_second_step_cuda, "thread_blocks",
                                                             kernel_arguments,
                                                             tuning_parameters_second, lang=language,
                                                             restrictions=constraints,
                                                             grid_div_x=[], iterations=3, answer=control_arguments,
                                                             verify=kernel.verify_second_step, atol=1.0e-03, quiet=True)
                results_second[blocks] = results
        except Exception as error:
            print(error)
    # Tuning totals
    combined_configurations = list()
    for index, configuration in enumerate(results_first):
        if configuration["thread_blocks"] == 1:
            combined_configurations.append([configuration])
            combined_configurations[index][0]["total"] = configuration["time"]
        else:
            combined_configurations.append([configuration, min(results_second[configuration["thread_blocks"]],
                                           key=lambda x: x["time"])])
            combined_configurations[index][0]["total"] = configuration["time"] \
                                                         + combined_configurations[index][1]["time"]
    return min(combined_configurations, key=lambda x: x[0]["total"])


def tune_medianofmedians_1D(input_size, step_size, language):
    # First kernel
    kernel = Statistics.MedianOfMedians1D(input_size, step_size)
    tuning_parameters_first = dict()
    tuning_parameters_first["type"] = ["float"]
    tuning_parameters_first["block_size_x"] = [x for x in range(1, step_size + 1)]
    data = numpy.random.randn(input_size).astype(numpy.float32)
    medians = numpy.zeros(math.ceil(input_size / step_size)).astype(numpy.float32)
    kernel_arguments = [data, medians]
    control_arguments = [None, numpy.asarray(kernel.generate_control_data_first_step(data))]
    results_first = dict()
    try:
        if language == "CUDA":
            results_first, platform = kernel_tuner.tune_kernel("compute_median_of_medians_" + str(step_size) + "_1D_first_step",
                                                               kernel.generate_first_step_cuda, [math.ceil(input_size / step_size)],
                                                               kernel_arguments, tuning_parameters_first, lang=language,
                                                               grid_div_x=[], iterations=3,
                                                               answer=control_arguments,
                                                               verify=kernel.verify_first_step,
                                                               atol=1.0e-06, quiet=True)
    except Exception as error:
        print(error)
    print(results_first)


if __name__ == "__main__":
    # Parse command line
    parser = argparse.ArgumentParser(description="AOFmcKT: AOFlagger many-core Kernels Tuner")
    parser.add_argument("--input_size", help="Input size.", required=True, type=int)
    parser.add_argument("--language", help="Language: CUDA or OpenCL.", choices=["CUDA", "OpenCL"], required=True)
    parser.add_argument("--tune_meanandstddev_1D", help="Tune mean and standard deviation 1D kernel.",
                        action="store_true")
    parser.add_argument("--tune_medianofmedians_1D", help="Tune median of medians 1D kernel.", action="store_true")
    parser.add_argument("--step_size", help="Step size for the median of medians.", type=int)
    arguments = parser.parse_args()
    # Tuning
    if arguments.tune_meanandstddev_1D is True:
        print(tune_meanandstddev_1D(arguments.input_size, arguments.language))
    elif arguments.tune_medianofmedians_1D is True:
        print(tune_medianofmedians_1D(arguments.input_size, arguments.step_size, arguments.language))
