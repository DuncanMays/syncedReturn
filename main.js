const tf = require("@tensorflow/tfjs-node");
const process = require('process');
const mnist = require('mnist');

const test = require('./test_suite.js');
// const workFn = require('./work_fn.js').work;
const workFn = require('./work_fn_with_benchmark_and_delay.js').work;

const central_model = make_model();

let central_parameters = marshal_parameters(central_model.getWeights())

// we now import testing data for performance evaluation
const testing_data = mnist.set(0, 10000).test;

// formatting testing data 
const testing_input = tf.tensor(testing_data.map(x => x.input));
const testing_output = tf.tensor(testing_data.map(x => x.output));

// stop training after this number of milliseconds
const run_time = 1000*60;
const num_workers_per_job = 5;
let compute;

// this function defines a new instantiation of the model we want to train
function make_model() {
  // defining the model to train
  const model = tf.sequential();

  model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
  model.add(tf.layers.dense({units: 20, activation: 'relu'}))
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

  model.compile({loss: tf.losses.meanSquaredError, metrics:['accuracy'], optimizer: tf.train.adam()});

  return model;
}
  
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

// aggregates parameters returned from worker
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
function get_param_performance(worker_params){
  // gets an instance of the model in question
  const model = make_model();
  // sets that model's parameters to the given parameter set
  model.setWeights(worker_params);
  // gets its performance on testing data
  const performance = model.evaluate(testing_input, testing_output);
  // evaluate returns an array of tensors, we want straight numbers, so we call arraySync on every element in performance and return the result
  return performance.map(x => x.arraySync());
}

// this function deploys a job that executes workFn in workers
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

  job.on('error', (err) => {console.log('there was an error: ', err);});

  job.on('console', (msg) => {console.log("worker "+msg.sliceIndex+" logged: "+msg.message);});

  job.on('result', (result) => {
    console.log("Got a result from worker", result.sliceNumber);
    return_objs.push(result.result);

    // this block of code tests the performance of the returned parameter set
    const worker_params = demarshall_parameters(result.result.params);
    const performance = get_param_performance(worker_params);
    console.log('the parameters returned from worker had a loss and accuracy of', performance);
  });

  job.on('noProgress', (e) => {
    console.log('noProgress');
    console.log(e.progressReports.last.value);
  });

  // aistensorflow/tfjs gives us tfjs
  // colab-data/mnist gives us MNIST, libpng/libpng is a dependancy of this
  // dcp-polyfills/polyfills is just generally a good thing to have, it polyfills a lot of JS things that aren't normally offered in worker
  job.requires('tlr-mnist-shard/mnist.js');
  // job.requires('dcp-polyfills/polyfills');
  job.requires('aistensorflow/tfjs');

  // this is needed to use webGL
  job.requirements.environment.offscreenCanvas = true;

  let results = await job.exec(compute.marketValue);

  return results;
}

async function main() {
  // the compute api
  await require('dcp-client').init(process.argv);
  compute = require('dcp/compute');

  let performance = central_model.evaluate(testing_input, testing_output);
  console.log("here is the model's loss and accuracy on testing data:", performance.map(x => x.arraySync()));

  const worker_params = await deploy_learning_job();

  // I wrote a function to return fake results so we don't have to wait for DCP every time we test aggregation
  // const worker_params = test.fake_results(5);

  const new_params = aggregate(worker_params)

  central_model.setWeights(new_params);

  performance = central_model.evaluate(testing_input, testing_output);
  console.log("here is the model's loss and accuracy on testing data:", performance.map(x => x.arraySync()));

  process.exit();
}

main();