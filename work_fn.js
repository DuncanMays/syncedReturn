const tf = require("@tensorflow/tfjs-node");

async function workFn(slice_input, shared_input) {

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

  await progress(0);

  // // makes sure the worker starts a second after deploy_time
  // const wait_time = 1000 + shared_input.deploy_time - Date.now();
  // if (wait_time > 0) {
  //   // waits for a given amount of time
  //   await new Promise((resolve, reject) => {
  //     setTimeout(() => {
  //       resolve();
  //     }, wait_time);
  //   })
  // }

  // in worker, console.log sends the given string back to the client and so is async
  await console.log('worker start');

  await progress(0.1);

  // the time that this worker will stop trianing and return to client
  const stop_time = shared_input.deploy_time + shared_input.run_time;

  // if stop time is less than 5 seconds in the future, then quit
  if (stop_time - Date.now() <= 5000){
    console.log('worker quit due to lack of time');

    const return_obj = {
      completed_batches: 0,
      params: null
    };

    return return_obj;

  } else{

    // imports the required modules
    tf = require('tfjs');
    tf.setBackend('cpu');
    await tf.ready();
    mnist = require('mnist.js');
    
    // we now define our model
    const model = tf.sequential();

    model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
    model.add(tf.layers.dense({units: 20, activation: 'relu'}))
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

    model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

    // we now set the weights of our model
    let params = demarshall_parameters(shared_input.params);
    model.setWeights(params);

    await console.log('starting download of mnist');

    // downloads MNIST shard 1 out of 12
    let data = await mnist.load(Math.round(12*Math.random()));
    await progress(0.15);

    // preprocesses data
    let imagesTensor = await tf.tensor2d(data.images, [Math.floor(data.images.length / 784), 784]);
    let labelsTensor = await tf.tensor2d(data.labels, [Math.floor(data.labels.length / 10), 10]);

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
            progress((0.95 - data_load_progress)*((Date.now() - shared_input.deploy_time)/shared_input.run_time) + data_load_progress);
          }
        }
      }
    });

    const return_obj = {
      completed_batches: completedBatches,
      params: marshal_parameters(model.getWeights())
    };

    return return_obj;
  }
}

module.exports = {work: workFn}