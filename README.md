# syncedReturn
A federated learning algorithm where system heterogeneity is addressed by allocating workers different amounts of data. The amount of data each worker should receive is calculated from benchmarking scores

## Overview
This project uses DCP, which is a task-sharing protocol written in JavaScript. It's meant to run in a compute group with local workers and two local servers which the workers pull slice data from. These two servers which provide slice input are: the IP_server, which simply serves workers their IP address, and the DA_server, which serves the amount of data each worker has been allocated, given their IP address. There are also two DCP jobs, one deployed by init.js pulls slice input from IP_server and runs benchmarks on all workers to calculate the amount of data each should be allocated. The second job pulls slice input from the DA server, downloads data, trains our model, and returns the trained parameters back to the client. The trained models from all workers are then aggregated in main.js and then redistributed for another round of training.

## File Guide

### data_requirements.js
Holds the names of the DCP modules which contain the MNIST data we will train on.

### DA_server.js
Reads data_allocation.txt and serves workers the integer representing the amount of data they've been allocated on port 8000.

### example.js
An example of using RemoteDataSet on DCP. This is the feature that allows workers to pull slice input from a URL.

### init.js
Runs a benchmarking job to determine hos much data each worker should be allocated.

### init_work_function.js
The worker function for init.js. Runs benchmarks and reports back to the client the amount of data the worker should be allocated and the worker's IP address.

### IP_server.js
Serves back the IP address of a GET request on port 7999.

### lint/
Contains all the old code that I'm not using anymore, explore at your own risk.

### lz_string.js
A compression library that has, itself, been compressed.

### main.js
The client for the job that performs distributed learning. Aggregates the trained models and runs diagnostics.

### main_work_function.js
Work function for main.js, downloads data to workers and trains a local model on that data. 

### save_model.js
Creates a dummy model and then saves it to a file out.txt which isn't uploaded to github. This only exists to get a measurement of the number of bytes the compressed model takes up. 

### sslcert/
Holds the SSL certification information

### tfjs_testing.js
A local playground for me to play with TFJS callbacks without having to wait for DCP workers to respond.