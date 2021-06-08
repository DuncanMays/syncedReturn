
wrk_fn = require('./work_function_1.js');
const data_requirements = require('./data_requirements.js');

// needed to collect command line arguements
const process = require('process');
const tf = require('@tensorflow/tfjs');

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

// returns the MNIST model that we will train here
function get_model() {
  const model = tf.sequential();

  model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
  model.add(tf.layers.dense({units: 20, activation: 'relu'}))
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

  model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

  return model;
}

const central_model = get_model();
const central_params = marshal_parameters(central_model.getWeights());

const NUM_SLICES = 3;
const SHARED_INPUT = {
  benchmark_length:100,
  deploy_time: Date.now(),
  time_for_training: 60000,
  show_logs: true,
  params: central_params
}

async function main() {
  await require('dcp-client').init(process.argv);
  const compute = require('dcp/compute');

  // creating slices
  slices = [];
  for (let i=0; i<NUM_SLICES; i++) {
    slices.push(i);
  }
  
  let job = compute.for(slices, SHARED_INPUT, wrk_fn);

  job.on('accepted', () => {console.log("Job accepted was accepted by the scheduler");});
  job.on('status', (status) => {console.log("Got a status update:", status);});
  job.on('result', (value) => console.log("Got a result:", value.result));
  job.on('error', (err) => {console.log('there was an error: ', err);});

  job.requires(data_requirements);
  job.requires('lzstring/lzstring');
  job.requires('lazy_loader/lazy_loader');
  job.requires('aistensorflow/tfjs');
  
  let results = await job.exec(0.01);

  process.exit()
}

main()
