import argparse
import numpy
import kernel_tuner

import Statistics


def tune_statistics_1D(input_size, language):
    # First kernel
    kernel = Statistics.Statistics1D(input_size)
    tuning_parameters_first = dict()
    tuning_parameters_first["type"] = ["float"]
    tuning_parameters_first["block_size_x"] = [2**x for x in range(5, 11)]
    tuning_parameters_first["items_per_thread"] = [2**x for x in range(0, 8)]
    tuning_parameters_first["thread_blocks"] = [2**x for x in range(0, 17)]
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
                results_first = kernel_tuner.tune_kernel("compute_statistics_1D_first_step",
                                                         kernel.generate_first_step_cuda, "thread_blocks",
                                                         kernel_arguments, tuning_parameters_first, lang=language,
                                                         restrictions=constraints, grid_div_x=[], iterations=3,
                                                         answer=control_arguments, verify=kernel.verify_first_step,
                                                         atol=1.0e-03)
    except Exception as error:
        print(error)
    # Second kernel
    tuning_parameters_second = dict()
    tuning_parameters_second["block_size_x"] = [2**x for x in range(1, 11)]
    tuning_parameters_second["thread_blocks"] = [1]
    results_second = dict()
    for blocks in tuning_parameters_first["thread_blocks"]:
        kernel = Statistics.Statistics1D(blocks * 3)
        triplets = numpy.random.random_integers(2, 2 + blocks, blocks * 3).astype(numpy.float32)
        statistics = numpy.zeros(2).astype(numpy.float32)
        kernel_arguments = [triplets, statistics]
        constraints = ["block_size_x <= " + str(blocks)]
        control_arguments = [None, numpy.asarray(kernel.generate_control_data_second_step(triplets))]
        try:
            if language == "CUDA":
                results = kernel_tuner.tune_kernel("compute_statistics_1D_second_step",
                                                   kernel.generate_second_step_cuda, "thread_blocks", kernel_arguments,
                                                   tuning_parameters_second, lang=language, restrictions=constraints,
                                                   grid_div_x=[], iterations=3, answer=control_arguments,
                                                   verify=kernel.verify_second_step, atol=1.0e-03)
                results_second[blocks] = results
        except Exception as error:
            print(error)
    # Tuning totals
    for index, configuration in enumerate(results_first):
        if configuration[index]["thread_blocks"] == 1:
            configuration[index]["total"] = configuration[index]["time"]
        else:
            configuration[index]["total"] = configuration[index]["time"] \
                                            + min(results_second[configuration[index]["thread_blocks"]],
                                                  key=lambda x: x["time"])


if __name__ == "__main__":
    # Parse command line
    parser = argparse.ArgumentParser(description="AOFmcKT: AOFlagger many-core Kernels Tuner")
    parser.add_argument("--input_size", help="Input size.", required=True, type=int)
    parser.add_argument("--language", help="Language: CUDA or OpenCL.", choices=["CUDA", "OpenCL"], required=True)
    parser.add_argument("--tune_statistics_1D", help="Tune \"compute_statistics_1D()\" kernel.",
                        action="store_true")
    arguments = parser.parse_args()
    # Tuning
    if arguments.tune_statistics_1D is True:
        tune_statistics_1D(arguments.input_size, arguments.language)
