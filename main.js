const tf = require("@tensorflow/tfjs-node");
const process = require('process');

const central_model = tf.sequential();

central_model.add(tf.layers.dense({units: 500, inputShape: [784], activation: 'relu'}))
central_model.add(tf.layers.dense({units: 200, activation: 'relu'}))
central_model.add(tf.layers.dense({units: 200, activation: 'relu'}))
central_model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

central_model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

let central_parameters = marshal_parameters(central_model.getWeights())
// stop training after this number of milliseconds
run_time = 1000*60;

// turns the parameter object that model.getWeights into an array that is JSON serializable
function marshal_parameters(param_tensor) {
  let params = param_tensor.map(x => x.arraySync());
  return params;
}

// turns the JSON serializable parameter array into a paramter object composed of tensors that can be passed to model.setWeights
function demarshall_parameters(param_array) {
  let params = param_array.map(x => tf.tensor(x));
  return params;
}

function progress(input) {
  console.log('progress ', input);
}

async function workFn(input) {
  // imports the required modules
  tf = require('tfjs');
  tf.setBackend('cpu');
  await tf.ready();
  mnist = require('mnist.js');

  await progress(0);

  // in worker, console.log sends the iven string back to the client and so is async
  await console.log('worker start');

  await progress(0.1);

  // the time that this worker will stop trianing and return to client
  const stop_time = input.deploy_time + input.run_time;


  // we now define our model
  const model = tf.sequential();

  model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
  model.add(tf.layers.dense({units: 20, activation: 'relu'}))
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

  model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

  await console.log('starting download of mnist');


  // downloads MNIST shard 1 out of 12
  let data = await mnist.load(1);
  await progress(0.15);

  // preprocesses data
  let imagesTensor = await tf.tensor2d( data.images, [ data.images.length / 784, 784] );
  let labelsTensor = await tf.tensor2d( data.labels, [ data.labels.length / 10, 10 ] );

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
          progress((0.95 - data_load_progress)*((Date.now() - input.deploy_time)/input.run_time) + data_load_progress);
        }
      }
    }});

  return input;
}

async function main() {
  // the compute api
  await require('dcp-client').init(process.argv);
  const compute = require('dcp/compute');

  const num_workers = 5;
  const deploy_time = Date.now();

  // each worker will be given an object that tells it when the job was deployed and how long since then to return
  let slices = [];
  let worker_input
  for (let i=0; i<num_workers; i++) {
    worker_input = {
      deploy_time: deploy_time,
      run_time: run_time
    };

    slices = slices.concat(worker_input);
  }
  
  let job = compute.for(slices, workFn);

  job.on('accepted', () => {console.log("Job accepted was accepted by the scheduler");});

  job.on('status', (status) => {console.log("Got a status update: ", status);});

  job.on('error', (err) => {console.log("there was an error: ", err);});

  job.on('console', (msg) => {console.log("a worker logged: ", msg.message);});

  job.on('result', (value) => console.log("Got a result: ", value.result));

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

  process.exit()
}

main();