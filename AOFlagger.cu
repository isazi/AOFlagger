// Plain templatization of Linus code, not changing it for now

#define TRUE 1
#define FALSE 0
#define BASE_SENSITIVITY 1.0f
#define MAX_ITERS  7
#define FIRST_THRESHOLD 6.0f
#define SIR_VALUE 0.4f

// Swap utility function
template<typename Type>
__global__ inline void swap(Type * array, unsigned int x, unsigned int y) {
    Type temp = array[x];
    array[x] = array[y];
    array[y] = temp;
}

// Sort an array in place
template<typename Type>
__device__ Type bitonic_sort(Type * values, int n, int nr_flagged) {
    const int tid = threadIdx.x;

    for ( int k = 2; k <= n; k *= 2) {
        for ( int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if ( ixj > tid ) {
                if ( (tid & k) == 0 ) {
                    if ( values[tid] > values[ixj] ) {
                        swap(values[tid], values[ixj]);
                    }
                } else {
                    if ( values[tid] < values[ixj] ) {
                        swap(values[tid], values[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }
    return values[nr_flagged + (n - nr_flagged) / 2];
}

template<typename Type>
__device__ Type sum_values(Type * values) {
    unsigned int tid = threadIdx.x;

    for ( unsigned int s = blockDim.x / 2; s > 32; s >>= 1 ) {
        if ( tid < s ) {
            values[tid] += values[tid + s];
        }
        __syncthreads();
    }

    if ( tid < 32 ) {
        values[tid] += values[tid + 32];
        values[tid] += values[tid + 16];
        values[tid] += values[tid + 8];
        values[tid] += values[tid + 4];
        values[tid] += values[tid + 2];
        values[tid] += values[tid + 1];
    }

    return values[0];
}

__device__ void count_flags(unsigned int * nr_flagged, LocalFlagsType * flags) {
    unsigned int tid = threadIdx.x;
    if ( flags[tid] == TRUE ) {
        atomicAdd(nr_flagged, 1);
    }
}

template<typename Type>
__device__ void sum_threshold(Type * values, LocalFlagsType * flags, float median, float stddev, int n) {
    int window = 1;
    int tid = threadIdx.x;
    float factor = stddev * BASE_SENSITIVITY;
    float sum;
    int pos;
    float threshold;

    for ( int i = 0; i < MAX_ITERS; i++ ) {
        threshold = fma((FIRST_THRESHOLD * powf(1.5f, i) / window), factor, median);
        sum = 0.0f;
        if ( tid % window == 0 ) {
            for ( pos = tid; pos < tid + window; pos++ ) {
                if ( flags[pos] != TRUE ) {
                    sum += values[pos];
                } else {
                    sum += threshold;
                }
            }
            if ( sum >= window * threshold ) {
                for ( pos = tid; pos < tid + window; pos++ ) {
                    flags[pos] = TRUE;
                }
            }
        }
        window *= 2;
    }
}

__global__ void sir_operator(LocalFlagsType * d_flags, int n) {
    LocalFlagsType * flags = &(d_flags[(blockIdx.x * n)]);
    float credit = 0.0f;
    float w;
    float max_credit0;

    for ( int i = 0; i < n; i++ ) {
        w = flags[i] ? SIR_VALUE : SIR_VALUE - 1.0f;
        max_credit0 = credit > 0.0f ? credit : 0.0f;
        credit = max_credit0 + w;
        flags[i] = credit >= 0.0f;
    }
    credit = 0;
    for ( int i = n - 1; i > 0; i-- ) {
        w = flags[i] ? SIR_VALUE : SIR_VALUE - 1.0f;
        max_credit0 = credit > 0.0f ? credit : 0.0f;
        credit = max_credit0 + w;
        flags[i] = credit >= 0.0f | flags[i];
    }
}

// MODIFIED, not equivalent to Linus code because our data structures are different
template<typename Type>
__global__ void reduce_freq(Type * values, Type * results, unsigned int number_of_channels,
    unsigned int number_of_samples) {
    extern __shared__ Type shared[];
    Type result = 0;

    // NOTE: Terrible memory access pattern
    for ( unsigned int channel = threadIdx.x; channel < number_of_channels; channel += blockDim.x) {
        result += values[(channel * number_of_samples) + blockIdx.x];
    }
    shared[threadIdx.x] = result;
    __syncthreads();
    result = sum_values(shared);
    if ( threadIdx.x == 0 ) {
        results[blockIdx.x] = result;
    }
}

template<typename Type>
__device__ void winsorize(Type * shared, int nr_flagged, int n) {
    if ( threadIdx.x < (0.1f * (n - nr_flagged) + nr_flagged) ) {
        shared[threadIdx.x] = shared[(int)(0.1f * (n - nr_flagged) + nr_flagged)];
    }
    if ( threadIdx.x > (0.9f * (n - nr_flagged) + nr_flagged) ) {
        shared[threadIdx.x] = shared[(int)(0.9f * (n - nr_flagged) + nr_flagged)];
    }
}

// MODIFIED, not equivalent to Linus code because our data structures are different
template<typename Type>
__global__ void reduce_time(Type * values, Type * results, unsigned int number_of_samples) {
    extern __shared__ Type shared[];
    Type result = 0;

    for ( unsigned int sample = threadIdx.x; sample < number_of_samples; sample++ ) {
        result += values[(blockIdx.x * number_of_samples) + sample];
    }
    shared[threadIdx.x] = result;
    __syncthreads();
    result = sum_values(shared);
    if ( threadIdx.x == 0 ) {
        results[blockDim.x] = result;
    }
}

// MODIFIED, not equivalent to Linus code because our data structures are different
template<typename Type>
__global__ void flagger_freq(Type * values, LocalFlagsType * global_flags, unsigned int * nr_flagged,
    unsigned int number_of_channels, unsigned int number_of_samples) {
    extern __shared__ Type shared[];
    LocalFlagsType * local_flags = (LocalFlagsType *) &(shared[number_of_channels]);
    unsigned int tid = threadIdx.x;
    Type median;
    Type stddev;

    // NOTE: terrible memory access pattern
    shared[tid] = values[(threadIdx.x * number_of_samples) + blockIdx.x];
    local_flags[tid] = 0;
    __syncthreads();

    for ( unsigned int i = 0; i < 2; i++ ) {
        Type sum = 0;
        Type squared_sum = 0;
        unsigned int local_nr_flagged = nr_flagged[blockIdx.x];

        median = bitonic_sort(shared, number_of_channels, local_nr_flagged);
        if ( tid >= local_nr_flagged ) {
            winsorize(shared, local_nr_flagged, number_of_channels);
        }
        __syncthreads();
        sum = sum_values(shared);

        // NOTE: terrible memory access pattern
        shared[tid] = values[(threadIdx.x * number_of_samples) + blockIdx.x];
        if ( local_flags[tid] ) {
            shared[tid] = 0;
        }
        __syncthreads();
        bitonic_sort(shared, number_of_channels, local_nr_flagged);
        if ( tid >= local_nr_flagged ) {
            winsorize(shared, local_nr_flagged, number_of_channels);
            shared[tid] *= shared[tid];
        }
        __syncthreads();
        squared_sum = sum_values(shared);
        stddev = sqrtf(squared_sum / number_of_channels - (sum / number_of_channels * sum / number_of_channels));
        // NOTE: terrible memory access pattern
        shared[tid] = values[(threadIdx.x * number_of_samples) + blockIdx.x];
        if ( local_flags[tid] ) {
            shared[tid] = 0;
        }
        __syncthreads();
        sum_threshold(shared, local_flags, median, stddev, number_of_channels);
        nr_flagged[blockIdx.x] = 0;
        count_flags(&(nr_flagged[blockIdx.x]), local_flags);
    }
    // NOTE: terrible memory access pattern
    global_flags[(threadIdx.x * number_of_samples) + blockIdx.x]
        = local_flags[tid] | global_flags[(threadIdx.x * number_of_samples) + blockIdx.x];
}

// MODIFIED, not equivalent to Linus code because our data structures are different
template<typename Type>
__global__ void flagger_time(Type * values, LocalFlagsType * global_flags, unsigned int * nr_flagged,
    unsigned int number_of_samples) {
    extern __shared__ Type shared[];
    LocalFlagsType * local_flags = (LocalFlagsType *) &(shared[number_of_samples]);
    unsigned int tid = threadIdx.x;
    Type median;
    Type stddev;

    shared[tid] = values[(blockIdx.x * number_of_samples) + threadIdx.x];
    local_flags[tid] = 0;
    __syncthreads();

    for ( unsigned int i = 0; i < 2; i++ ) {
        Type sum = 0;
        Type squared_sum = 0;
        unsigned int local_nr_flagged = nr_flagged[blockIdx.x];

        median = bitonic_sort(shared, number_of_samples, local_nr_flagged);
        if ( tid >= local_nr_flagged ) {
            winsorize(shared, local_nr_flagged, number_of_samples);
        }
        __syncthreads();
        sum = sum_values(shared);

        shared[tid] = values[(blockIdx.x * number_of_samples) + threadIdx.x];
        if ( local_flags[tid] ) {
            shared[tid] = 0;
        }
        __syncthreads();
        bitonic_sort(shared, number_of_samples, local_nr_flagged);
        if ( tid >= local_nr_flagged ) {
            winsorize(shared, local_nr_flagged, number_of_samples);
            shared[tid] *= shared[tid];
        }
        __syncthreads();
        squared_sum = sum_values(shared);
        stddev = sqrtf(squared_sum / number_of_samples - (sum / number_of_samples * sum / number_of_samples));
        shared[tid] = values[(blockIdx.x * number_of_samples) + threadIdx.x];
        if ( local_flags[tid] ) {
            shared[tid] = 0;
        }
        __syncthreads();
        sum_threshold(shared, local_flags, median, stddev, number_of_samples);
        nr_flagged[blockIdx.x] = 0;
        count_flags(&(nr_flagged[blockIdx.x]), local_flags);
    }
    global_flags[(blockIdx.x * number_of_samples) + threadIdx.x]
        = local_flags[tid] | global_flags[(blockIdx.x * number_of_samples) + threadIdx.x];
}

