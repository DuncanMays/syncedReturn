const tf = require("@tensorflow/tfjs");
const fs = require('fs');

// ***************************************************************************************************************************
// these functions exist for testing purposes and will not be exported

function progress(i) {
  console.log('Progress: '+i);
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

// returns the MNIST model that we will train here
  function get_model() {
    const model = tf.sequential();

    model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
    model.add(tf.layers.dense({units: 20, activation: 'relu'}))
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

    model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

    return model;
  }

// ***************************************************************************************************************************

async function workFn(shared_input) {

  // imports the required modules
  tf = require('tfjs');
  tf.setBackend('cpu');
  await tf.ready();

  function log(payload) {
    if (shared_input.show_logs) {
      console.log(payload);
    }
  }

  log(tf.version);

  const { lazy_load } = require('lazy_loader');
  const lz = require('lzstring');

  // the progress after we've loaded the data
  const DATA_LOAD_PROGRESS = 0.1;
  // the progress after we've trained the model
  const TRAIN_PROGRESS = 0.95;

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

  // lets the scheduler, and client, know that the worker has started
  await progress(0);

  // the number of data points that the average worker can train the model on per second
  const sample_rate = 3763.0546790022713;
  // const sample_rate = Math.round(Math.random()*1000);

  // the backlog and delay times, randomly determined and in seconds
  const backlog_time  = Math.round(Math.random()*6);
  const delay_time  = Math.round(Math.random()*6);

  // we now wait for backlog_time + delay_time seconds;
  await new Promise((resolve, reject) => {
    setTimeout(() => {
      resolve();
    }, 1000*(backlog_time + delay_time));
  })


  // the time that the worker has left to train
  const remaining_time = shared_input.deploy_time + shared_input.time_for_training - Date.now()

  // if we have no remaining time, or if we have less than ten seconds, simply return a null indicator that the client will ignore
  if (remaining_time <= 10000) {
    log('started too late');
    return 'null'
  }

  // the number of samples that the worker should download, maxed at 60,000 since that's the maximum number of data points in MNIST
  const num_samples_to_download = Math.min(sample_rate*remaining_time, 60000);

  // since MNIST is sharded into bundles of 500 data points, the number of shards to download is the number of samples to download divided by 500
  num_data_shards = Math.round(num_samples_to_download/500);

  log('number of data shards to download: ' + num_data_shards.toString());

  // this is a list of the index of all the shards, from which we will randomly select the indices of the shards that we will download
  all_indices = [];
  for (let i=1; i<=120; i++) {
    all_indices.push(i);
  }

  // this is a list of the indices of the shards that we will download
  shard_indices = [];
  for (let j=0; j<num_data_shards; j++) {
    // select a random element in all_indices
    index_index = Math.round(Math.random()*(all_indices.length-1));
    // append that element to shard_indices
    shard_indices.push(all_indices[index_index]);
    // remove it from all_indicies so that it will not be selected twice
    all_indices = all_indices.slice(0, index_index).concat(all_indices.slice(index_index+1, all_indices.length))
  }  

  await progress(DATA_LOAD_PROGRESS/2);

  // we now download the data

  let x_data_raw = [];
  let y_data_raw = [];

  for (let k=0; k<shard_indices.length; k++) {
    let index = shard_indices[k];

    // in worker this will be a call to require, here we just load the data out of files I've got on disk
    // let raw = JSON.parse(fs.readFileSync('/home/duncan/Documents/DCL/sharding_mnist/shards500/train_shard_'+index).toString());

    // downloads and decompresses data
    await lazy_load(['train_shard_'+index]);
    mnist_shard = require('train_shard_'+index);

    raw = JSON.parse(lz.decompressFromBase64(mnist_shard));

    x_data_raw = x_data_raw.concat(raw.images);
    y_data_raw = y_data_raw.concat(raw.labels);
  }

  const imagesTensor = tf.tensor(x_data_raw).reshape([-1, 28*28]);
  const labelsTensor = tf.oneHot(y_data_raw, 10);

  progress(DATA_LOAD_PROGRESS)

  // we now train for the remaining time 

  // defining our model and setting the parameters
  const model = get_model();
  const params = demarshall_parameters(shared_input.params);
  model.setWeights(params);

  // these variables will be used during training
  let completedBatches = 0;
  let current_remaining_time;
  let time_left_for_training;

  log('starting training')

  await model.fit(imagesTensor, labelsTensor, {
    yieldEvery: 5000,
    epochs: 100,
    callbacks: {
      onBatchEnd: (batch, logs) => {
          completedBatches = completedBatches + 1;
      },

      onYield: (epoch, batch, logs) => {

        current_remaining_time = shared_input.deploy_time + shared_input.time_for_training - Date.now();
        progress_ratio = 1 - current_remaining_time/remaining_time;

        if (progress_ratio >= 1) {
          // stops training if the time limit is exceeded
          log('training time exceeded');
          model.stopTraining = true;
        } else {
          progress((TRAIN_PROGRESS-DATA_LOAD_PROGRESS)*progress_ratio + DATA_LOAD_PROGRESS);
        }
      }
    }
  });

  progress(TRAIN_PROGRESS)

  const return_obj = {
    completed_batches: completedBatches,
    params: marshal_parameters(model.getWeights())
  };

  return return_obj;

  progress(1);

  return return_obj;
}

module.exports = workFn