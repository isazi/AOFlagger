
import math
import numpy


class Statistics1D:
    input_size = int()
    configuration = dict()

    CUDA_TEMPLATE = """__global__ void compute_statistics_1D(const <%TYPE%> * const input_data, 
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

    def __init__(self, size):
        self.input_size = size

    # Generate CUDA code
    def generate_cuda(self, configuration):
        self.configuration = configuration
        code = Statistics1D.CUDA_TEMPLATE.replace("<%TYPE%>", configuration["type"])
        code = code.replace("<%THREADS_PER_BLOCK%>", str(configuration["block_size_x"]))
        code = code.replace("<%ITEMS_PER_BLOCK%>", str(math.ceil(self.input_size
                                                                 / int(configuration["thread_blocks"]))))
        code = code.replace("<%ITEMS_PER_ITERATION%>", str(int(configuration["block_size_x"])
                            * int(configuration["items_per_thread"])))
        code = code.replace("<%THREADS_PER_BLOCK_HALVED%>", str(int(int(configuration["block_size_x"]) / 2)))
        local_variables = str()
        local_compute = str()
        for item in range(0, int(configuration["items_per_thread"])):
            local_variables = local_variables + Statistics1D.LOCAL_VARIABLES.replace("<%ITEM_NUMBER%>", str(item))
            if self.input_size % \
                    (int(configuration["thread_blocks"]) * int(configuration["block_size_x"])
                     * int(configuration["items_per_thread"])) == 0:
                local_compute = local_compute + Statistics1D.LOCAL_COMPUTE_NOCHECK.replace("<%ITEM_NUMBER%>", str(item))
            else:
                local_compute = local_compute + Statistics1D.LOCAL_COMPUTE_CHECK.replace("<%ITEM_NUMBER%>", str(item))
                local_compute = local_compute.replace("<%ITEMS_PER_BLOCK%>", str(math.ceil(self.input_size
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
                thread_reduce = thread_reduce + Statistics1D.THREAD_REDUCE.replace("<%ITEM_NUMBER%>", str(item))
            code = code.replace("<%THREAD_REDUCE%>", thread_reduce)
        else:
            code = code.replace("<%THREAD_REDUCE%>", "")
        return code

    # Generate OpenCL code
    def generate_opencl(self, configuration):
        self.configuration = configuration
        code = str()
        return code

    def verify(self, control_data, data, atol=None):
        counter = 0.0
        mean = 0.0
        variance = 0.0
        for item in range(0, self.configuration["thread_blocks"] * 3, 3):
            temp = data[item + 1] - mean
            mean = ((counter * mean) + (data[item] * data[item + 1])) / (counter + data[item])
            variance = variance + data[item + 2] + ((temp * temp) * ((counter * data[item]) / (counter + data[item])))
            counter = counter + data[item]
        variance = variance / (counter - 1)
        result = numpy.allclose(control_data, [counter, mean, variance], atol)
        if result is False:
            numpy.set_printoptions(precision=6, suppress=True)
            print(control_data)
            print(data[0:(self.configuration["thread_blocks"] * 3)])
            print([counter, mean, variance])
        return result
