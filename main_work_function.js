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

    model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
    model.add(tf.layers.dense({units: 20, activation: 'relu'}))
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

    model.compile({loss: tf.losses.softmaxCrossEntropy, metrics:['accuracy'], optimizer: tf.train.adam()});

    return model;
  }

// ***************************************************************************************************************************

async function workFn(slice_input, shared_input) {

  // imports the required modules
  tf = require('tfjs');
  tf.setBackend('cpu');
  await tf.ready();

  function log(payload) {
    if (shared_input.show_logs) {
      console.log(payload);
    }
  }

  const { lazy_load } = require('lazy_loader');
  const lz = require('lzstring');

  // the progress after we've loaded the data
  const DATA_LOAD_PROGRESS = 0.1;
  // the progress after we've trained the model
  const TRAIN_PROGRESS = 0.95;

  // turns the parameter object that model.getWeights from an array of tensors into an array or arrays so that it is JSON serializable
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

    model.compile({loss: tf.losses.softmaxCrossEntropy, metrics:[], optimizer: tf.train.adam()});

    return model;
  }

  // since MNIST is sharded into bundles of 500 data points, the max number of shards is 120
  let num_data_shards = Math.min(slice_input, 120);

  log('number of data shards to download: ' + num_data_shards);

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

  const num_epochs = 1

  await model.fit(imagesTensor, labelsTensor, {
    epochs: num_epochs,
    callbacks: {
      onBatchEnd: (batch, logs) => {
          completedBatches = completedBatches + 1;

          // give the scheduler a progress update every 100 batches 
          if (completedBatches%100 == 0) {
            train_progress_ratio = completedBatches*32/(imagesTensor.shape[0] * num_epochs);
            progress((TRAIN_PROGRESS-DATA_LOAD_PROGRESS)*train_progress_ratio + DATA_LOAD_PROGRESS);
          }
      }
    }
  });

  // model.fit(imagesTensor, labelsTensor, {
  //   batchSize: imagesTensor.shape[0],
  //   epochs: num_epochs,
  //   yieldEvery: 1000,
  //   callbacks: {
  //     onYield: (epoch, batch, logs) => {
  //       train_progress_ratio = batch*32/(imagesTensor.shape[0] * num_epochs);
  //       progress((TRAIN_PROGRESS-DATA_LOAD_PROGRESS)*train_progress_ratio + DATA_LOAD_PROGRESS);
  //     }
  //   }
  // });

  progress(TRAIN_PROGRESS)

  const return_obj = {
    num_shards: num_data_shards,
    completed_batches: completedBatches,
    params: marshal_parameters(model.getWeights())
  };

  tf.dispose(model);
  tf.dispose(imagesTensor);
  tf.dispose(labelsTensor);

  progress(1);

  return return_obj;
}

module.exports = workFn
