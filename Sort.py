
import numpy

class BubbleSortSharedMemory1D:
    input_size = int()

    CUDA_TEMPLATE = """__global__ void sharedmemory_sort_1D(const <%TYPE%> * const input_data, <%TYPE%> * const output_data) {
        __shared__ <%TYPE%> local_data[<%INPUT_SIZE%>];
        
        // Load data in shared memory
        for ( unsigned int item = threadIdx.x; item < <%INPUT_SIZE%>; item += <%THREADS_PER_BLOCK%> ) {
            local_data[item] = input_data[item];
        }
        __syncthreads();
        // Sort data
        for ( unsigned int step = 0; step < <%STEPS%>; step++ ) {
            if ( (threadIdx.x % 2) == (step % 2) ) {
                for ( unsigned int item = threadIdx.x; item < <%INPUT_SIZE%> - 1; item += <%THREADS_PER_BLOCK%> ) {
                    if ( local_data[item] > local_data[item + 1] ) {
                        <%TYPE%> temp = local_data[item];
                        local_data[item] = local_data[item + 1];
                        local_data[item + 1] = temp;
                    }
                }
            }
            __syncthreads();
        }
        // Store sorted output
        for ( unsigned int item = threadIdx.x; item < <%INPUT_SIZE%>; item += <%THREADS_PER_BLOCK%> ) {
            output_data[item] = local_data[item];
        }
    }"""

    def __init__(self, input_size):
        self.input_size = input_size

    def generate_cuda(self, configuration):
        code = self.CUDA_TEMPLATE.replace("<%INPUT_SIZE%>", str(self.input_size))
        code = code.replace("<%TYPE%>", configuration["type"])
        code = code.replace("<%STEPS%>", str(self.input_size))
        code = code.replace("<%THREADS_PER_BLOCK%>", str(configuration["block_size_x"]))
        return code

    @staticmethod
    def verify(control_data, data, atol=None):
        result = numpy.allclose(control_data, data, atol)
        if result is False:
            numpy.set_printoptions(precision=6, suppress=True)
            print(control_data)
            print(data)
        return result

class BitonicSortSharedMemory1D:
    input_size = int()

    CUDA_TEMPLATE = """__global__ void bitonic_sort_1D(const <%TYPE%> * const input_data, <%TYPE%> * const output_data) {
        __shared__ <%TYPE%> local_data[<%INPUT_SIZE%>];
        
        // Load data in shared memory
        for ( unsigned int item = threadIdx.x; item < <%INPUT_SIZE%>; item += <%THREADS_PER_BLOCK%> ) {
            local_data[item] = input_data[item];
        }
        __syncthreads();
        // Sort data
        // Bitonic sort by Linus Schoemaker
        for ( unsigned int k = 2; k <= <%INPUT_SIZE%>; k *= 2) {
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
        // Store sorted output
        for ( unsigned int item = threadIdx.x; item < <%INPUT_SIZE%>; item += <%THREADS_PER_BLOCK%> ) {
            output_data[item] = local_data[item];
        }
        }"""

    def __init__(self, input_size):
        self.input_size = input_size

    def generate_cuda(self, configuration):
        """
        Generate CUDA code for 1D bitonic sort in shared memory.

        :param configuration: kernel configuration
        :return: generated CUDA code.
        """
        code = self.CUDA_TEMPLATE.replace("<%INPUT_SIZE%>", str(self.input_size))
        code = code.replace("<%TYPE%>", configuration["type"])
        code = code.replace("<%THREADS_PER_BLOCK%>", str(configuration["block_size_x"]))
        return code

    @staticmethod
    def verify(control_data, data, atol=None):
        result = numpy.allclose(control_data, data, atol)
        if result is False:
            numpy.set_printoptions(precision=6, suppress=True)
            print(control_data)
            print(data)
        return result
