
#include "AOFlagger.h"

__global__ void swap(float * data, unsigned int x, unsigned int y) {
    float temp = data[x];
    data[x] = data[y];
    data[y] = temp;
}