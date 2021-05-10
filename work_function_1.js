const tf = require("@tensorflow/tfjs");
const fs = require('fs');

function progress(i) {
  console.log('Progress: '+i);
}

async function workFn(slice_input, shared_input) {

  // imports the required modules
  // tf = require('tfjs');
  // tf.setBackend('webgl');
  // await tf.ready();
  // mnist = require('mnist.js');

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

  // returns the number of samples per second that the worker is able to train the model on
  const NUM_DUMMY_SAMPLES = shared_input.benchmark_length;
  async function benchmark(){
    const dummy_model = get_model()

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
    const sample_rate = total/num_epochs;

    return sample_rate
  }

  await progress(0);

  // the number of data points that we can train the model on per second
  // const sample_rate = await benchmark();
  const sample_rate = Math.round(Math.random()*1000);

  // the backlog and delay times, randomly determined and in seconds
  const backlog_time  = Math.round(Math.random()*6);
  const delay_time  = Math.round(Math.random()*6);

  // the time that the worker has left to train
  const remaining_time = shared_input.time_for_training - backlog_time - delay_time;

  // the number of samples that the worker should download, maxed at 60,000 since that's the maximum number of data points in MNIST
  const num_samples_to_download = Math.min(sample_rate*remaining_time, 60000);

  // since MNIST is sharded into bundles of 500 data points, the number of shards to download is the number of samples to download divided by 500
  num_data_shards = Math.round(num_samples_to_download/500);

  // this is a list of the index of all the shards, from which we will randomly select the indices of the shards that we will download
  all_indices = [];
  for (let i=1; i<=120; i++) {
    all_indices.push(i);
  }

  // this is a list of the indices of the shards that we will downlod=ad
  shard_indices = [];
  for (let j=0; j<num_data_shards; j++) {
    // select a random element in all_indices
    index_index = Math.round(Math.random()*(all_indices.length-1));
    // append that element to shard_indices
    shard_indices.push(all_indices[index_index]);
    // remove it from all_indicies so that it will not be selected twice
    all_indices = all_indices.slice(0, index_index).concat(all_indices.slice(index_index+1, all_indices.length))
  }  

  await progress(0.05);

  // we now download the data

  let x_data_raw = [];
  let y_data_raw = [];

  for (let k=0; k<shard_indices.length; k++) {
    let index = shard_indices[k];

    // in worker this will be a call to require
    let raw = JSON.parse(fs.readFileSync('/home/duncan/Documents/DCL/sharding_mnist/shards500/train_shard_'+index).toString());

    x_data_raw = x_data_raw.concat(raw.images);
    y_data_raw = y_data_raw.concat(raw.labels);
  }

  const imagesTensor = tf.tensor(x_data_raw).reshape([-1, 28*28]);
  const labelsTensor = tf.oneHot(y_data_raw, 10);

  console.log(imagesTensor.shape);
  console.log(labelsTensor.shape);

}

module.exports = workFn