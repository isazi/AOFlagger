
import kernel_tuner
import numpy

FILENAME = "AOFlagger.cu"

with open(FILENAME) as file:
    kernel_source = file.read()
