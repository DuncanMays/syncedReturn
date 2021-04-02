// this program deploys a job on DCP that does 2 things:
// it runs a benchmark on the worker
// it trains a model on MNIST and calculates the sample rate of the model, which is the number of samples per second the model is able to train on
// it will then return these two figures to the client
// we're doing this because we want to understand the releationship between the benchmark and the sample rate of the model
// this will help us determine how much data to download to worker for a model that we want to train for a certain amount of time, based on the benchmarking score

// where the results returned from each slice will be stored
const all_results = []

// this function exists so that we can run the work function locally without crashing
// progress is defined in worker, but not in node
function progress(input) {
  console.log(input);
}

// this function will be sent to workers on DCP, it will download data, train a model on that data while recording the speed of training, and it will run a benchmark
async function workFn(input) {
  // imports the required modules
  const tf = require('tfjs');
  tf.setBackend('webgl');
  await tf.ready();
  const mnist = require('mnist.js');

  const NUM_DUMMY_SAMPLES = 500

  progress(0);

  async function benchmark(){
    const dummy_model = tf.sequential();

    dummy_model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
    dummy_model.add(tf.layers.dense({units: 20, activation: 'relu'}))
    dummy_model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

    dummy_model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

    x_dummy = tf.randomUniform([NUM_DUMMY_SAMPLES, 784])
    y_dummy = tf.randomUniform([NUM_DUMMY_SAMPLES, 10])

    // these variables are needed to keep track of the model's sample rate
    let epoch_start = 0;
    let epoch_end = 0;
    let time_for_epoch, samples_per_second;
    const rate_data = [];
    const num_epochs = 10

    // records the start of training
    epoch_start = Date.now();

    await dummy_model.fit(x_dummy, y_dummy, {
      epochs: num_epochs,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          // records the end of the epoch
          epoch_end = Date.now();

          // calculates the average number of samples trained every second during the last epoch
          // this is calculated from the number of samples in each epoch (x_dummy.shape[0]) and the amount of time the epoch took
          time_for_epoch = epoch_end - epoch_start;
          samples_per_second = 1000*x_dummy.shape[0]/time_for_epoch;

          // pushes the training rate to a list
          rate_data.push(samples_per_second)

          // this callback runs in between epochs, and so the end of this function is the start of the next epoch
          epoch_start = Date.now();
        }
      }
    });

    // we will now take the average of the samples_per_second of the epochs
    total = 0
    for (let i=0; i<rate_data.length; i++) {
    total = total + rate_data[i];
    }
    sample_rate = total/num_epochs;

    return sample_rate
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

  // records the start of training
  epoch_start = Date.now();

  await model.fit(imagesTensor, labelsTensor, {
    epochs: num_epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        // records the end of the epoch
        epoch_end = Date.now();

        // calculates the average number of samples trained every second during the last epoch
        // this is calculated from the number of samples in each epoch (imagesTensor.shape[0]) and the amount of time the epoch took
        time_for_epoch = epoch_end - epoch_start;
        samples_per_second = 1000*imagesTensor.shape[0]/time_for_epoch;

        // pushes the training rate to a list
        rate_data.push(samples_per_second)

        // reports progress to the scheduler
        const prog = (1-data_load_progress)*epoch/num_epochs + data_load_progress;
        progress(prog);

        // this callback runs in between epochs, and so the end of this function is the start of the next epoch
        epoch_start = Date.now();
      }
    }
  });

  // we will now take the average of the samples_per_second of the epochs
  total = 0
  for (let i=0; i<rate_data.length; i++) {
    total = total + rate_data[i];
  }
  sample_rate = total/num_epochs;

  // runs a benchmark and return the anverage sample rate with the benchmarking score
  const return_obj = {
    sample_rate: sample_rate,
    benchmark_score: await benchmark()
  };

  return return_obj;
}

// this function exists solely to test the results saving mechanism, writing to files in node is not something I've got a lot of experience with, and so I made this function to play around
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

  // adds listenners to log status updates, errors, and console logs from worker
  job.on('accepted', () => {console.log("Job accepted was accepted by the scheduler!");});
  job.on('status', (status) => {console.log("Got a status update: ", status);});
  job.on('error', (err) => {console.log('there was an error: ', err);});
  job.on('console', (msg) => {console.log("worker "+msg.sliceIndex+" logged: "+msg.message);});

  // if a worker returns a result, push it to the list all_results so we can record it at the end of the program
  job.on('result', (result) => {
    console.log("Got a result from worker", result.sliceNumber);
    all_results.push(result.result);
  });

  // if a worker dies from no progress, we need to know the last reported progress value so we know where the error was in the work fn
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

// this function exists for the purpose of reporting fake results to test the recording system
// that way we can test the file writing systems without having to wait for DCP
function deploy_fake_job(num_slices) {
  const result_array = [];
  for (let i=0; i<num_slices; i++) {
    result_array.push(test_wrk_fn(1));
  }
  const result_obj = {};
  result_obj.values = () => {return result_array}
  return result_obj;
}

// saves results to a file
function save_results() {
  console.log('saving results to head.data');
  const fs = require('fs');
  fs.writeFileSync('./head.data', JSON.stringify(all_results));
  process.exit();
}
process.on('SIGINT', save_results);

async function main() {
  // the compute api
  await require('dcp-client').init(process.argv);
  compute = require('dcp/compute');

  const results = await deploy_job(3);
  
  save_results();

  process.exit();
}

main()