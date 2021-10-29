// other files
const wrk_fn = require('./main_work_function.js');
const lz = require('./lz_string.js');
const data_requirements = require('./data_requirements.js');

// node modules
const process = require('process');
const mnist = require('mnist');
const tf = require('@tensorflow/tfjs');
const fs = require('fs');

// we now import testing data for performance evaluation
const testing_data = mnist.set(0, 10000).test;

// formatting testing data 
const testing_input = tf.tensor(testing_data.map(x => x.input));
const testing_output = tf.tensor(testing_data.map(x => x.output));

/// turns the parameter object that model.getWeights from an array of tensors into an array or arrays so that it is JSON serializable
function marshal_parameters(param_tensor) {
  let params = param_tensor.map(x => x.arraySync());
  return lz.compressToBase64(JSON.stringify(params));
}

// turns the JSON serializable array of arrays into an array of tensors that can be passed to model.setWeights
function demarshall_parameters(param_array) {
  param_array = JSON.parse(lz.decompressFromBase64(param_array));
  let params = param_array.map(x => tf.tensor(x));
  return params;
}

// returns the MNIST model that we will train here
function get_model() {
  const model = tf.sequential();

  model.add(tf.layers.dense({units: 300, inputShape: [784], activation: 'relu'}))
  model.add(tf.layers.dense({units: 124, activation: 'relu'}))
  model.add(tf.layers.dense({units: 60, activation: 'relu'}))
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

  model.compile({loss: tf.losses.softmaxCrossEntropy, metrics:['accuracy'], optimizer: tf.train.adam()});

  return model;
}

// aggregates parameters returned from worker
function aggregate(parameter_array) {
  console.log('aggregating');

  parameter_array = parameter_array.filter((p) => {return p.params})

  // splitting the input into the parameter sets and the weights for each parameter set
  let params = parameter_array.map(x => demarshall_parameters(x.params));
  let weights = parameter_array.map(x => x.num_shards);

  // normalizing the weights
  let sum = 0;
  weights.map((x) => {sum = sum + x});
  weights = weights.map(x => x/sum);

  // multiplying the parameters by their weights
  params = params.map((element, index) => {
    weight = weights[index];
    // remember that each parameter set is an array of tensors representing the parameters of each layer
    element = element.map(tensor => tensor.mul(weight));
    return element;
  });

  // summing the arrays of tensors
  let new_params = [];
  for (let i=0; i<params[0].length; i++) {
    let summands = [];
    for (let j=0; j<params.length; j++) {
      summands.push(params[j][i]);
    }
    new_params.push(tf.stack(summands).sum(dim=0));
  }

  return new_params
}

// returns the performance of a set of paramters on testing data
// calls evaluate on a model with the given parameters, and so will return an array containing the loss and any metrics
function get_param_performance(params){
  // gets an instance of the model in question
  const model = get_model();
  // sets that model's parameters to the given parameter set
  model.setWeights(params);
  // gets its performance on testing data
  const performance = model.evaluate(testing_input, testing_output);
  // evaluate returns an array of tensors, we want straight numbers, so we call arraySync on every element in performance and return the result
  return performance.map(x => x.arraySync());
}

const central_model = get_model();
let central_params = marshal_parameters(central_model.getWeights());

let performance = central_model.evaluate(testing_input, testing_output);
console.log("here is the model's loss and accuracy on testing data on initialization:", performance.map(x => x.arraySync()));

// these symbols need to be global
let compute, RemoteDataSet;

const NUM_SLICES = 5;
const SHARED_INPUT = {
  deploy_time: Date.now(),
  time_for_training: 1000*60,
  show_logs: false,
  params: central_params
}

// creating slice_inputs
let slice_inputs = [];
const slice_URL = 'https://192.168.2.19:8000'
for (let i=0; i<NUM_SLICES; i++) {
  slice_inputs.push(slice_URL);
}

const worker_params = [];
let total_completed_batches = 0;
let total_shards_downloaded = 0;

async function deploy_job() {

  let job = compute.for(slice_inputs, wrk_fn, [SHARED_INPUT]);

  job.on('accepted', () => {console.log("Job accepted was accepted by the scheduler", job.address.slice(0, 15));});
  job.on('status', (status) => {console.log("Got a status update:", status);});
  job.on('error', (err) => {console.log('there was an error:', err);});
  job.on('console', (msg) => {console.log("worker "+msg+" logged: "+msg.message);});

  job.on('result', (value) => {
    console.log("Got a result from worker", value.sliceNumber);

    // workers will return 'null' if they can't train for some reason - this could be due to time constraints, low benchmarking score, etc.
    if (value.result != 'null') {
      worker_params.push(value.result);
      total_completed_batches = total_completed_batches + value.result.completed_batches;
      total_shards_downloaded = total_shards_downloaded + value.result.num_shards;
    }
  });

  job.requires(data_requirements);
  job.requires('lzstring/lzstring');
  job.requires('lazy_loader/lazy_loader');
  job.requires('aistensorflow/tfjs');

  job.computeGroups = [{joinKey: 'queens-edge', joinSecret: 'P8PuQ0oCXm'}]
  
  console.log('deploying job')
  let results = await job.exec(0.001);

  console.log('complete learning job');
  job.cancel()
}

function assess_parameters(params) {
  const model = get_model();

  model.setWeights(params);

  performance = model.evaluate(testing_input, testing_output);

  return performance.map(x => x.arraySync());
}

function assess_results() {
  console.log('assessing results');

  trained_params = aggregate(worker_params);

  central_model.setWeights(trained_params);

  performance = central_model.evaluate(testing_input, testing_output);

  [loss, accuracy] = performance.map(x => x.arraySync());
  
  console.log('training time: ', SHARED_INPUT.time_for_training/1000)

  console.log("testing loss:", loss);
  console.log("testing accuracy:", accuracy);
  console.log("total completed batches:", total_completed_batches);
  console.log("total shards downloaded:", total_shards_downloaded);

  return {'acc': accuracy, 'loss': loss};
}

async function main() {
  for (let i=0; i<10; i++) {
    await deploy_job();

    for (let j=0; j<worker_params.length; j++) {
      console.log('assessing parameters from worker:', j);

      params = demarshall_parameters(worker_params[j].params);
      [loss, accuracy] = assess_parameters(params);

      console.log("testing loss:", loss);
      console.log("testing accuracy:", accuracy);
    }

    scores = assess_results();

    SHARED_INPUT.params = marshal_parameters(central_model.getWeights());
  }

  process.exit();
}

require('dcp-client').init(process.argv).then(() => {
  compute = require('dcp/compute');

  slice_inputs = new compute.RemoteDataSet(slice_inputs)

  main();
});