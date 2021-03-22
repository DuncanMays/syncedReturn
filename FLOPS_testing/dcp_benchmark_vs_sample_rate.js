// this program deploys a job on DCP that does 2 things:
// it runs a benchmark on the worker
// it trains a model on MNIST and calculates the sample rate of the model, which is the number of samples per second the model is able to train on
// it will then return these two figures to the client
// we're doing this because we want to understand the releationship between the benchmark and the sample rate of the model
// this will help us determine how much data to download to each model, based on the benchmarking score

const all_results = []

function progress(input) {
  console.log(input);
}

async function workFn(input) {
  // imports the required modules
  tf = require('tfjs');
  tf.setBackend('cpu');
  await tf.ready();
  mnist = require('mnist.js');

  progress(0);

  function benchmark(){
    // this is the width of every matrix being multiplied, larger n values will give bigger rates, because GPUs perform matrix computations in parallel
    // depending on the texture size of the GPU (I think), large n values will crach some computers
    // this is the largest n value I've experimented with that doesn't cause a lot of crashes, the next highest was 1024 so mayeb this is worth experimenting with again.
    let n = 512;
    let iters = 50;

    // it's important to instantiate and multiply the tensors at least once before the benchmarking begins
    // this is because there is some setup tfjs has to do behind the scenes, which will take a constant amount of time and scew the results of our benchmark
    // doing a mult before the benchmakring loop will perform this setup, and since we're using the same tensors in the loop, the setup will not have to be done twice
    var matrix1 = tf.randomNormal([n,n], dtype='float32');
    var matrix2 = tf.randomNormal([n,n], dtype='float32');
    var matrixOut = tf.matMul(matrix1, matrix2);
    var outSync = matrixOut.dataSync();

    let start = Date.now();

    for (let i=0;i<iters;i++){
        var matrixOut = tf.matMul(matrix1, matrix2);
        var outSync = matrixOut.dataSync();
        matrixOut.dispose();
    };

    let end = Date.now();

    matrix1.dispose();
    matrix2.dispose();

    let ops = (n**3) + ((n-1)*(n**2)); // n^2*(n-1) additions and n^3 mults
    let elapsed = end-start;

    let rate = iters*ops/elapsed/1e6;

    return rate
  }

  // we now define our model
  const model = tf.sequential();

  model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
  model.add(tf.layers.dense({units: 20, activation: 'relu'}))
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

  model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

  // downloads MNIST shard 1 out of 12
  let data = await mnist.load(Math.round(12*Math.random()));
  progress(0.15);

  // preprocesses data
  let imagesTensor = await tf.tensor2d(data.images, [Math.floor(data.images.length / 784), 784]);
  let labelsTensor = await tf.tensor2d(data.labels, [Math.floor(data.labels.length / 10), 10]);

  // updates progress, we need to keep track of the progress at this point since the training loop below needs to know at what progress level to start at
  const data_load_progress = 0.2;
  progress(data_load_progress);

  // these variables are needed to keep track of the model's sample rate
  let epoch_start = 0;
  let epoch_end = 0;
  let time_for_epoch, samples_per_second;
  const rate_data = [];
  const num_epochs = 10

  epoch_start = Date.now();

  await model.fit(imagesTensor, labelsTensor, {
    epochs: num_epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        epoch_end = Date.now();

        time_for_epoch = epoch_end - epoch_start;
        samples_per_second = 1000*imagesTensor.shape[0]/time_for_epoch;

        rate_data.push(samples_per_second)

        epoch_start = Date.now();

        const prog = (1-data_load_progress)*epoch/num_epochs + data_load_progress;
        progress(prog);
      }
    }
  });

  total = 0
  for (let i=0; i<rate_data.length; i++) {
    total = total + rate_data[i];
  }
  sample_rate = total/num_epochs;

  const return_obj = {
    sample_rate: sample_rate,
    benchmark_score: benchmark()
  };

  return return_obj;
}

function test_wrk_fn(input) {
  progress(1);

  const return_obj = {
    sample_rate: 5,
    benchmark_score: 25
  };

  return return_obj;
}

// this function deploys a job that executes workFn in workers
async function deploy_job(num_slices) {

  console.log('deploying job');

  // each worker will be given an object that tells it when the job was deployed and how long since then to return
  let slices = [];
  let worker_input
  for (let i=0; i<num_slices; i++) {
    let slice_input = i
    slices = slices.concat(slice_input);
  }
  
  // I have no idea why the last parameter, shared input, needs to be in an array, all I know is that this works
  let job = compute.for(slices, workFn);
  // let job = compute.for(slices, test_wrk_fn);

  job.on('accepted', () => {console.log("Job accepted was accepted by the scheduler");});

  job.on('status', (status) => {console.log("Got a status update: ", status);});

  job.on('error', (err) => {console.log('there was an error: ', err);});

  job.on('console', (msg) => {console.log("worker "+msg.sliceIndex+" logged: "+msg.message);});

  job.on('result', (result) => {
    console.log("Got a result from worker", result.sliceNumber);
    all_results.push(result.result);
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

  let results = await job.exec(0.002);
  // let results = await job.localExec();


  return results;
}

function deploy_fake_job(num_slices) {
  const result_array = [];
  for (let i=0; i<num_slices; i++) {
    result_array.push(test_wrk_fn(1));
  }
  const result_obj = {};
  result_obj.values = () => {return result_array}
  return result_obj;
}

function save_results() {
  console.log('saving results to results.data');
  const fs = require('fs');
  fs.writeFileSync('./results.data', JSON.stringify(all_results));
  process.exit();
}
process.on('SIGINT', save_results);

async function main() {
  // the compute api
  await require('dcp-client').init(process.argv);
  compute = require('dcp/compute');

  const results = await deploy_job(50);
  
  save_results();

  process.exit();
}

main()