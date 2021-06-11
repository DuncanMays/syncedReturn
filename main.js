
// wrk_fn = require('./work_function_1.js');
// wrk_fn = require('./work_function_2.js');
wrk_fn = require('./work_function_3.js');

const data_requirements = require('./data_requirements.js');

// needed to collect command line arguements
const process = require('process');
const mnist = require('mnist');
const tf = require('@tensorflow/tfjs');

// we now import testing data for performance evaluation
const testing_data = mnist.set(0, 10000).test;

// formatting testing data 
const testing_input = tf.tensor(testing_data.map(x => x.input));
const testing_output = tf.tensor(testing_data.map(x => x.output));

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

  model.compile({loss: tf.losses.meanSquaredError, metrics:['MSE', 'accuracy'], optimizer: tf.train.adam()});

  return model;
}

// aggregates parameters returned from worker
function aggregate(parameter_array) {
  console.log('aggregating');

  parameter_array = parameter_array.filter((p) => {return p.params})

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
const central_params = marshal_parameters(central_model.getWeights());

let performance = central_model.evaluate(testing_input, testing_output);
console.log("here is the model's loss and accuracy on testing data on initialization:", performance.map(x => x.arraySync()));

const NUM_SLICES = 5;
const SHARED_INPUT = {
  benchmark_length:100,
  deploy_time: Date.now(),
  time_for_training: 3*60000,
  show_logs: false,
  params: central_params
}

const worker_params = [];

async function main() {

  await require('dcp-client').init(process.argv);
  const compute = require('dcp/compute');

  // creating slices
  slices = [];
  for (let i=0; i<NUM_SLICES; i++) {
    slices.push(SHARED_INPUT);
  }
  
  let job = compute.for(slices, wrk_fn);

  job.on('accepted', () => {console.log("Job accepted was accepted by the scheduler");});
  job.on('status', (status) => {console.log("Got a status update:", status);});
  job.on('error', (err) => {console.log('there was an error: ', err);});
  job.on('console', (msg) => {console.log("worker "+msg.sliceIndex+" logged: "+msg.message);});

  job.on('result', (value) => {
    console.log("Got a result from worker", value.sliceNumber);
    worker_params.push(value.result);
  });

  job.requires(data_requirements);
  job.requires('lzstring/lzstring');
  job.requires('lazy_loader/lazy_loader');
  job.requires('aistensorflow/tfjs');

  job.public.name = 'Federated Learning';
  
  let results = await job.exec(0.01);

  finish();
}

function finish() {
  console.log('wrapping up');

  trained_params = aggregate(worker_params);

  central_model.setWeights(trained_params);

  performance = central_model.evaluate(testing_input, testing_output);
  console.log("here is the model's loss and accuracy on testing data:", performance.map(x => x.arraySync()));

  process.exit();
}
process.on('SIGINT', finish);

// stops the program if it runs for longer that 1.5 time the training time
setTimeout(finish, 1.5*SHARED_INPUT.time_for_training);

main();
