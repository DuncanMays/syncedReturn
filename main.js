const tf = require("@tensorflow/tfjs-node");
const process = require('process');

const central_model = tf.sequential();

central_model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
central_model.add(tf.layers.dense({units: 20, activation: 'relu'}))
central_model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

central_model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

let central_parameters = marshal_parameters(central_model.getWeights())
// stop training after this number of milliseconds
const run_time = 1000*60;
const num_workers_per_job = 5;
let compute;

// turns the parameter object that model.getWeights from an array of tensors into an array or arrays so that it is JSON serializable
function marshal_parameters(param_tensor) {
  let params = param_tensor.map(x => x.arraySync());
  return params;
}

// turns the JSON serializable array of arrays into an array of tensors that can be passed to model.setWeights
function demarshall_parameters(param_array) {
  let params = param_array.map(x => tf.tensor(x));
  return params;
}

function progress(input) {
  console.log('progress ', input);
}

async function workFn(slice_input, shared_input) {
  // imports the required modules
  tf = require('tfjs');
  tf.setBackend('cpu');
  await tf.ready();
  mnist = require('mnist.js');

  // turns the parameter object that model.getWeights from an array of tensors into an array or arrays so that it is JSON serializable
  function marshal_parameters(param_tensor) {
    let params = param_tensor.map(x => x.arraySync());
    return params;
  }

  // turns the JSON serializable array of arrays into an array of tensors that can be passed to model.setWeights
  function demarshall_parameters(param_array) {
    let params = param_array.map(x => tf.tensor(x));
    return params;
  }

  await progress(0);

  // // makes sure the worker starts a second after deploy_time
  // const wait_time = 1000 + shared_input.deploy_time - Date.now();
  // if (wait_time > 0) {
  //   // waits for a given amount of time
  //   await new Promise((resolve, reject) => {
  //     setTimeout(() => {
  //       resolve();
  //     }, wait_time);
  //   })
  // }

  // in worker, console.log sends the given string back to the client and so is async
  await console.log('worker start');

  await progress(0.1);

  // the time that this worker will stop trianing and return to client
  const stop_time = shared_input.deploy_time + shared_input.run_time;

  // we now define our model
  const model = tf.sequential();

  model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
  model.add(tf.layers.dense({units: 20, activation: 'relu'}))
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

  model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

  // we now set the weights of our model
  let params = demarshall_parameters(shared_input.params);
  model.setWeights(params);

  await console.log('starting download of mnist');

  // downloads MNIST shard 1 out of 12
  let data = await mnist.load(1);
  await progress(0.15);

  // preprocesses data
  let imagesTensor = await tf.tensor2d(data.images, [data.images.length / 784, 784]);
  let labelsTensor = await tf.tensor2d(data.labels, [data.labels.length / 10, 10]);

  const data_load_progress = 0.2;
  await progress(data_load_progress);

  await console.log('starting training');

  let completedBatches = 0;
  await model.fit(imagesTensor, labelsTensor, {
    yieldEvery: 5000,
    epochs: 100,
    callbacks: {
      onBatchEnd: (batch, logs) => {
          completedBatches = completedBatches + 1;
      },
      onYield: (epoch, batch, logs) => {
        if (Date.now() > stop_time) {
          // stops training if the time limit is exceeded
          console.log('training time exceeded');
          model.stopTraining = true;
        } else {
          progress((0.95 - data_load_progress)*((Date.now() - shared_input.deploy_time)/shared_input.run_time) + data_load_progress);
        }
      }
    }
  });

  const return_obj = {
    completed_batches: completedBatches,
    params: marshal_parameters(model.getWeights())
  };

  return return_obj;
}

function aggregate(parameter_array) {
  // splitting the input into the parameter sets and the weights for each parameter set
  let params = parameter_array.map(x => demarshall_parameters(x.params));
  let weights = parameter_array.map(x => x.completed_batches);

  // normalizing the weights
  let sum = 0;
  weights.map((x) => {sum = sum + x});
  weights = weights.map(x => x/sum);

  // multiplying the parameters by the weights
  params = params.map((element, index) => {
    weight = weights[index];
    // remember that each parameter set is an array of tensors representing the parameters of each layer
    element = element.map(tensor => tensor.mul(weight));
  });

  new_params = params[0];
  for (let i=1; i<params.length; i++) {
    summand = params[i];
  }
}

async function deploy_learning_job() {
  const deploy_time = Date.now();
  let return_objs = [];

  // each worker will be given an object that tells it when the job was deployed and how long since then to return
  let slices = [];
  let worker_input
  for (let i=0; i<num_workers_per_job; i++) {
    let slice_input = {
      slice_number: i
    };

    slices = slices.concat(slice_input);
  }

  let shared_input = {
    deploy_time: deploy_time,
    run_time: run_time,
    params: central_parameters
  };
  
  // I have no idea why the last parameter, shared input, needs to be in an array, all I know is that this works
  let job = compute.for(slices, workFn, [shared_input]);

  job.on('accepted', () => {console.log("Job accepted was accepted by the scheduler");});

  job.on('status', (status) => {console.log("Got a status update: ", status);});

  job.on('error', (err) => {console.log("there was an error: ", err);});

  job.on('console', (msg) => {console.log("a worker logged: ", msg.message);});

  job.on('result', (result) => {
    console.log("Got a result from worker", result.sliceNumber);
    return_objs = return_objs.concat(result.result);
  });

  job.on('noProgress', (e) => {
    console.log('noProgress');
    console.log(e.progressReports.last.value);
  });

  // aistensorflow/tfjs gives us tfjs
  // colab-data/mnist gives us MNIST, libpng/libpng is a dependancy of this
  // dcp-polyfills/polyfills is just generally a good thing to have, it polyfills a lot of JS things that aren't normally offered in worker
  job.requires('aitf-mnist-shard/mnist.js');
  // job.requires('dcp-polyfills/polyfills');
  job.requires('aistensorflow/tfjs');

  // this is needed to use webGL
  // job.requirements.environment.offscreenCanvas = true;

  let results = await job.exec(compute.marketValue);

  aggregate(return_objs);
}

async function main() {
  // the compute api
  await require('dcp-client').init(process.argv);
  compute = require('dcp/compute');

  await deploy_learning_job();  

  process.exit();
}

main();