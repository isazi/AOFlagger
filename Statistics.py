import math
import numpy


class MeanAndStandardDeviation1D:
    input_size = int()
    configuration_first = dict()

    CUDA_TEMPLATE_FIRST = """__global__ void compute_statistics_1D_first_step(const <%TYPE%> * const input_data, 
            float * const statistics) {
        <%LOCAL_VARIABLES%>
        float temp;
        __shared__ float reduction_counter[<%THREADS_PER_BLOCK%>];
        __shared__ float reduction_mean[<%THREADS_PER_BLOCK%>];
        __shared__ float reduction_variance[<%THREADS_PER_BLOCK%>];
        
        for ( unsigned int value_id = (blockIdx.x * <%ITEMS_PER_BLOCK%>) + threadIdx.x; 
                value_id < ((blockIdx.x + 1) * <%ITEMS_PER_BLOCK%>); 
                value_id += <%ITEMS_PER_ITERATION%> ) {
            <%TYPE%> value;
            
            <%LOCAL_COMPUTE%>
        }
        <%THREAD_REDUCE%>
        reduction_counter[threadIdx.x] = counter_0;
        reduction_mean[threadIdx.x] = mean_0;
        reduction_variance[threadIdx.x] = variance_0;
        __syncthreads();
        unsigned int threshold = <%THREADS_PER_BLOCK_HALVED%>;
        for ( unsigned int value_id = threadIdx.x; threshold > 0; threshold /= 2 ) {
            if ( value_id < threshold ) {
                temp = reduction_mean[value_id + threshold] - mean_0;
                mean_0 = ((counter_0 * mean_0) + (reduction_counter[value_id + threshold] 
                    * reduction_mean[value_id + threshold])) 
                    / (counter_0 + reduction_counter[value_id + threshold]);
                variance_0 += reduction_variance[value_id + threshold] + ((temp * temp) 
                    * ((counter_0 * reduction_counter[value_id + threshold]) 
                    / (counter_0 + reduction_counter[value_id + threshold])));
                counter_0 += reduction_counter[value_id + threshold];
                reduction_mean[threadIdx.x] = mean_0;
                reduction_variance[threadIdx.x] = variance_0;
                reduction_counter[threadIdx.x] = counter_0;
            }
            __syncthreads();
        }
        if ( threadIdx.x == 0 ) {
            statistics[(blockIdx.x * 3)] = counter_0;
            statistics[(blockIdx.x * 3) + 1] = mean_0;
            statistics[(blockIdx.x * 3) + 2] = variance_0;
        }
    }"""
    LOCAL_VARIABLES = """float counter_<%ITEM_NUMBER%> = 0;
        float mean_<%ITEM_NUMBER%> = 0;
        float variance_<%ITEM_NUMBER%> = 0;
    """
    LOCAL_COMPUTE_NOCHECK = """value = input_data[value_id + <%ITEM_OFFSET%>];
            counter_<%ITEM_NUMBER%> += 1;
            temp = value - mean_<%ITEM_NUMBER%>;
            mean_<%ITEM_NUMBER%> += temp / counter_<%ITEM_NUMBER%>;
            variance_<%ITEM_NUMBER%> += temp * (value - mean_<%ITEM_NUMBER%>);
    """
    LOCAL_COMPUTE_CHECK = """if ( (value_id + <%ITEM_OFFSET%> < <%INPUT_SIZE%>)
                && (value_id + <%ITEM_OFFSET%> < ((blockIdx.x + 1) * <%ITEMS_PER_BLOCK%>)) ) {
                value = input_data[value_id + <%ITEM_OFFSET%>];
                counter_<%ITEM_NUMBER%> += 1;
                temp = value - mean_<%ITEM_NUMBER%>;
                mean_<%ITEM_NUMBER%> += temp / counter_<%ITEM_NUMBER%>;
                variance_<%ITEM_NUMBER%> += temp * (value - mean_<%ITEM_NUMBER%>);
            }
    """
    THREAD_REDUCE = """temp = mean_<%ITEM_NUMBER%> - mean_0;
        mean_0 = ((counter_0 * mean_0) + (counter_<%ITEM_NUMBER%> * mean_<%ITEM_NUMBER%>)) 
            / (counter_0 + counter_<%ITEM_NUMBER%>);
        variance_0 += variance_<%ITEM_NUMBER%> + ((temp * temp) * ((counter_0 * counter_<%ITEM_NUMBER%>) 
            / (counter_0 + counter_<%ITEM_NUMBER%>)));
        counter_0 += counter_<%ITEM_NUMBER%>;
    """
    CUDA_TEMPLATE_SECOND = """__global__ void compute_statistics_1D_second_step(const float3 * const triplets, 
                float * const statistics) {
            float temp;
            float counter_0 = 0.0f;
            float mean_0 = 0.0f;
            float variance_0 = 0.0f;
            __shared__ float reduction_counter[<%THREADS_PER_BLOCK%>];
            __shared__ float reduction_mean[<%THREADS_PER_BLOCK%>];
            __shared__ float reduction_variance[<%THREADS_PER_BLOCK%>];

            for ( unsigned int value_id = threadIdx.x; value_id < <%ITEMS_PER_BLOCK%>; 
                value_id += <%THREADS_PER_BLOCK%> ) {
                float3 triplet = triplets[value_id];
                
                temp = triplet.y - mean_0;
                mean_0 = ((counter_0 * mean_0) + (triplet.x * triplet.y)) / (counter_0 + triplet.x);
                variance_0 += triplet.z + ((temp * temp) * ((counter_0 * triplet.x) / (counter_0 + triplet.x)));
                counter_0 += triplet.x;
            }
            reduction_counter[threadIdx.x] = counter_0;
            reduction_mean[threadIdx.x] = mean_0;
            reduction_variance[threadIdx.x] = variance_0;
            __syncthreads();
            unsigned int threshold = <%THREADS_PER_BLOCK_HALVED%>;
            for ( unsigned int value_id = threadIdx.x; threshold > 0; threshold /= 2 ) {
                if ( value_id < threshold ) {
                    temp = reduction_mean[value_id + threshold] - mean_0;
                    mean_0 = ((counter_0 * mean_0) + (reduction_counter[value_id + threshold] 
                        * reduction_mean[value_id + threshold])) 
                        / (counter_0 + reduction_counter[value_id + threshold]);
                    variance_0 += reduction_variance[value_id + threshold] + ((temp * temp) 
                        * ((counter_0 * reduction_counter[value_id + threshold]) 
                        / (counter_0 + reduction_counter[value_id + threshold])));
                    counter_0 += reduction_counter[value_id + threshold];
                    reduction_mean[threadIdx.x] = mean_0;
                    reduction_variance[threadIdx.x] = variance_0;
                    reduction_counter[threadIdx.x] = counter_0;
                }
                __syncthreads();
            }
            if ( threadIdx.x == 0 ) {
                statistics[blockIdx.x] = mean_0;
                statistics[blockIdx.x + 1] = sqrt(variance_0 / (counter_0 - 1));
            }
        }"""

    def __init__(self, size):
        self.input_size = size

    # Generate CUDA code for first step
    def generate_first_step_cuda(self, configuration):
        self.configuration_first = configuration
        code = MeanAndStandardDeviation1D.CUDA_TEMPLATE_FIRST.replace("<%TYPE%>", configuration["type"])
        code = code.replace("<%THREADS_PER_BLOCK%>", str(configuration["block_size_x"]))
        code = code.replace("<%ITEMS_PER_BLOCK%>", str(math.ceil(self.input_size
                                                                 / int(configuration["thread_blocks"]))))
        code = code.replace("<%ITEMS_PER_ITERATION%>", str(int(configuration["block_size_x"])
                                                           * int(configuration["items_per_thread"])))
        code = code.replace("<%THREADS_PER_BLOCK_HALVED%>", str(int(int(configuration["block_size_x"]) / 2)))
        local_variables = str()
        local_compute = str()
        for item in range(0, int(configuration["items_per_thread"])):
            local_variables = local_variables + MeanAndStandardDeviation1D.LOCAL_VARIABLES.replace("<%ITEM_NUMBER%>", str(item))
            if self.input_size % \
                    (int(configuration["thread_blocks"]) * int(configuration["block_size_x"])
                     * int(configuration["items_per_thread"])) == 0:
                local_compute = local_compute + MeanAndStandardDeviation1D.LOCAL_COMPUTE_NOCHECK.replace("<%ITEM_NUMBER%>", str(item))
            else:
                local_compute = local_compute + MeanAndStandardDeviation1D.LOCAL_COMPUTE_CHECK.replace("<%ITEM_NUMBER%>", str(item))
                local_compute = local_compute.replace("<%ITEMS_PER_BLOCK%>",
                                                      str(math.ceil(self.input_size
                                                                    / int(configuration["thread_blocks"]))))
                local_compute = local_compute.replace("<%INPUT_SIZE%>", str(self.input_size))
            if item == 0:
                local_compute = local_compute.replace(" + <%ITEM_OFFSET%>", "")
            else:
                local_compute = local_compute.replace("<%ITEM_OFFSET%>",
                                                      str(item * int(configuration["block_size_x"])))
        code = code.replace("<%LOCAL_VARIABLES%>", local_variables)
        code = code.replace("<%LOCAL_COMPUTE%>", local_compute)
        if int(configuration["items_per_thread"]) > 1:
            thread_reduce = str()
            for item in range(1, int(configuration["items_per_thread"])):
                thread_reduce = thread_reduce + MeanAndStandardDeviation1D.THREAD_REDUCE.replace("<%ITEM_NUMBER%>", str(item))
            code = code.replace("<%THREAD_REDUCE%>", thread_reduce)
        else:
            code = code.replace("<%THREAD_REDUCE%>", "")
        return code

    # Generate CUDA code for second step
    def generate_second_step_cuda(self, configuration):
        code = MeanAndStandardDeviation1D.CUDA_TEMPLATE_SECOND.replace("<%THREADS_PER_BLOCK%>", str(configuration["block_size_x"]))
        code = code.replace("<%ITEMS_PER_BLOCK%>", str(self.input_size))
        code = code.replace("<%THREADS_PER_BLOCK_HALVED%>", str(int(int(configuration["block_size_x"]) / 2)))
        return code

    # Generate OpenCL code for first step
    def generate_first_step_opencl(self, configuration):
        # TODO: implement the method
        self.configuration_first = configuration
        code = str()
        return code

    # Generate OpenCL code for second step
    def generate_second_step_opencl(self, configuration):
        # TODO: implement the method
        self.configuration_first = configuration
        code = str()
        return code

    def verify_first_step(self, control_data, data, atol=None):
        counter = 0.0
        mean = 0.0
        variance = 0.0
        for item in range(0, self.configuration_first["thread_blocks"] * 3, 3):
            temp = data[item + 1] - mean
            mean = ((counter * mean) + (data[item] * data[item + 1])) / (counter + data[item])
            variance = variance + data[item + 2] + ((temp * temp) * ((counter * data[item]) / (counter + data[item])))
            counter = counter + data[item]
        variance = variance / (counter - 1)
        result = numpy.allclose(control_data, [counter, mean, variance], atol)
        if result is False:
            numpy.set_printoptions(precision=6, suppress=True)
            print(control_data)
            print(data[0:(self.configuration_first["thread_blocks"] * 3)])
            print([counter, mean, variance])
        return result

    def generate_control_data_second_step(self, data):
        counter = 0.0
        mean = 0.0
        variance = 0.0
        for item in range(0, self.input_size, 3):
            temp = data[item + 1] - mean
            mean = ((counter * mean) + (data[item] * data[item + 1])) / (counter + data[item])
            variance = variance + data[item + 2] + ((temp * temp) * ((counter * data[item]) / (counter + data[item])))
            counter = counter + data[item]
        variance = variance / (counter - 1)
        return [mean, math.sqrt(variance)]

    @staticmethod
    def verify_second_step(control_data, data, atol=None):
        result = numpy.allclose(control_data, data, atol)
        if result is False:
            numpy.set_printoptions(precision=6, suppress=True)
            print(control_data)
            print(data)
        return result

class MedianOfMedians1D:
    input_size = int()
    step_size = int()
    configuration_first = dict()

    CUDA_TEMPLATE_FIRST_STEP = """__global__ void compute_median_of_medians_<%STEP_SIZE%>_1D_first_step(const <%TYPE%> * const input_data, 
            float * const medians) {
        unsigned int first_global_item = blockIdx.x * <%STEP_SIZE%>;
        __shared__ <%TYPE%> local_data[<%STEP_SIZE%>];
        
        // Load data in shared memory
        for ( unsigned int item = threadIdx.x; item < <%STEP_SIZE%>; item += <%THREADS_PER_BLOCK%> ) {
            local_data[item] = input_data[first_global_item + item];
        }
        __syncthreads();
        // Sort data
        // Bitonic sort by Linus Schoemaker
        for ( unsigned int k = 2; k <= <%STEP_SIZE%>; k *= 2) {
            for ( unsigned int j = k / 2; j > 0; j /= 2) {
                unsigned int ixj = threadIdx.x ^ j;
                if ( ixj > threadIdx.x ) {
                    if ( (threadIdx.x & k) == 0 ) {
                        if ( local_data[threadIdx.x] > local_data[ixj] ) {
                            <%TYPE%> temp = local_data[threadIdx.x];
                            local_data[threadIdx.x] = local_data[ixj];
                            local_data[ixj] = temp;
                        }
                    } else {
                        if ( local_data[threadIdx.x] < local_data[ixj] ) {
                            <%TYPE%> temp = local_data[threadIdx.x];
                            local_data[threadIdx.x] = local_data[ixj];
                            local_data[ixj] = temp;
                        }
                    }
                }
                __syncthreads();
            }
        }
        // Store median
        if ( threadIdx.x == 0 ) {
            medians[blockIdx.x] = local_data[<%HALF_STEP%>];
        }
        }"""

    def generate_first_step_cuda(self, configuration):
        """
        Generate CUDA code for the first step.

        :param configuration: kernel configuration
        :return: generated CUDA code.
        """
        code = self.CUDA_TEMPLATE_FIRST_STEP.replace("<%STEP_SIZE%>", str(self.step_size))
        code = code.replace("<%TYPE%>", configuration["type"])
        code = code.replace("<%THREADS_PER_BLOCK%>", str(configuration["block_size_x"]))
        code = code.replace("<%HALF_STEP%>", str(math.floor(self.step_size / 2)))
        return code

    def __init__(self, input_size, step_size):
        """
        Constructor.

        :param input_size: size of the input array
        :param step_size: size of the step of which compute the real median
        """
        self.input_size = input_size
        self.step_size = step_size

    def generate_control_data_first_step(self, data):
        """
        Compute the first partial array of medians.

        :param data: a numpy array containing the input data
        :return: a numpy array containing the medians
        """
        # Divide input in blocks of size 'step_size'
        blocks = [data[i:i+self.step_size] for i in range(0, self.input_size, self.step_size)]
        # Compute and store the median of each block
        medians_of_blocks = list()
        for block in blocks:
            medians_of_blocks.append(sorted(block)[math.floor(len(block) / 2)])
        return medians_of_blocks

    @staticmethod
    def verify_first_step(control_data, data, atol=None):
        result = numpy.allclose(control_data, data, atol)
        if result is False:
            numpy.set_printoptions(precision=6, suppress=True)
            print(control_data)
            print(data)
        return result

    def generate_control_data_second_step(self, data):
        """
        Compute the median of medians.

        :param data: a numpy array containing the input data
        :return: the computed median of medians
        """
        medians_of_blocks = self.generate_control_data_first_step(data)
        # Compute and return median of medians
        return sorted(medians_of_blocks)[math.floor(len(medians_of_blocks) / 2)]

    @staticmethod
    def verify_second_step(control_data, data, atol=None):
        result = numpy.isclose(control_data, data, atol)
        if result is False:
            numpy.set_printoptions(precision=6, suppress=True)
            print(control_data)
            print(data)
        return result
