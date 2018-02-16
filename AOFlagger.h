#define TRUE 1
#define FALSE 0
#define BASE_SENSITIVITY 1.0f
#define MAX_ITERS  7
#define FIRST_THRESHOLD 6.0f
#define SIR_VALUE 0.4f

// Swap
__global__ void swap(float * data, unsigned int x, unsigned int y);
// Sort an array in place
__global__ void bitonic_sort(float * values, unsigned int n, unsigned int nr_flagged);
// Reduce an array
__global__ void sum_values(float * values);
// Count the flagged values
__global__ void count_flags(unsigned int * nr_flagged, float * flags);
// Sum Threshold
__global__ void sum_threshold(float * values, float * flags, float median, float stddev, int n);
// SIR operator
__global__ void sir_operator(float * d_flags, int n);
// Integrate over frequencies
__global__ void reduce_freq(float * values, float * results, unsigned int number_of_channels,
    unsigned int number_of_samples);
// Winsorize operator
__device__ void winsorize(float * shared, int nr_flagged, int n);
// Integrate over time
__global__ void reduce_time(float * values, float * results, unsigned int number_of_samples);
// Flag over frequencies
__global__ void flagger_freq(float * values, float * global_flags, unsigned int * nr_flagged,
    unsigned int number_of_channels, unsigned int number_of_samples);
// Flag over time
__global__ void flagger_time(float * values, float * global_flags, unsigned int * nr_flagged,
    unsigned int number_of_samples);