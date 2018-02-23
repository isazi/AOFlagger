# AOFlagger

AOFlagger is a Radio Frequency Interference (RFI) mitigation algorithm proposed by A. Offringa in the paper "[A morphological algorithm for improving radio-frequency interference detection](https://www.aanda.org/index.php?option=com_article&amp;access=doi&amp;doi=10.1051/0004-6361/201118497&amp;Itemid=129)".

In this repository I am toying around with some a real-time many-core implementation of said algorithm, using as a starting point the code developed by [Rob van Nieuwpoort and Linus Schoemaker](https://arxiv.org/pdf/1701.08197v1.pdf).

## Sequential Code Analysis

### Data

The input is a time series.
Data could be one or two dimensional, in the scenario that we are currently investigating.
The output is a flagging mask that, when applied to the data, cancels the RFI out.

### Statistics

The most important building block of the flagger are the statistics.
The statistics used in the code are:
- **Mean**
- **Median**
- **Standard Deviation**
- **Median Absolute Deviation**

Plus the *winsorized* version of mean and median.
Winsorized version of mean and median are still mean and median, but computed only on the middle part of the data.
In the original source code, the middle 80% of data is used, i.e. top and bottom 10% of values are excluded from the computation.

Of all the statistics, the most expensive to compute is the median; this is, assuming the input data are not already ordered.
Computing mean and standard deviation can be done in O(n) on the input data, without sorting; it is also possible to easily parallelize the computation of mean and standard deviation.