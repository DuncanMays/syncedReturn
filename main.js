const tf = require("@tensorflow/tfjs-node");
const process = require('process');

const test = require('./test_suite.js');
const workFn = require('./work_fn.js').work;

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
    return_objs.push(result.result);
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

  return results;

}

async function main() {
  // the compute api
  await require('dcp-client').init(process.argv);
  compute = require('dcp/compute');

  const worker_params = await deploy_learning_job();

  // I wrote a function to return fake results so we don't have to wait for DCP every time we test aggregation
  // const worker_params = test.fake_results(5);

  const new_params = aggregate(worker_params)

  central_model.setWeights(new_params);

  process.exit();
}

main();