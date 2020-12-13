const tf = require("@tensorflow/tfjs-node");

async function workFn(slice_input, shared_input) {
  // imports the required modules
  tf = require('tfjs');
  tf.setBackend('cpu');
  await tf.ready();
  mnist = require('mnist.js');

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

  await progress(0);

  console.log('running benchmark');
  const rate = benchmark();

  await progress(0.1);

  let imagesTensor;
  let labelsTensor;

  console.log('starting download of mnist');
  if (rate < 0.5) {
    console.log('worker identified as weak, with rate:', rate);

    // delays worker for 5 seconds
    await new Promise((resolve, reject) => {
      setTimeout(() => {
        resolve();
      }, 5000);
    })

    // only download 5000 data points
    let data = await mnist.load(Math.round(12*Math.random()));

    // preprocesses data
    imagesTensor = await tf.tensor2d(data.images, [Math.floor(data.images.length / 784), 784]);
    labelsTensor = await tf.tensor2d(data.labels, [Math.floor(data.labels.length / 10), 10]);

  } else {
    console.log('worker identified as strong, with rate:', rate);

    // download 10000 data points

    // randomly select 2 indices between 0 and 12
    const first_index = Math.round(12*Math.random());
    let second_index;
    for(let i=0; i<100; i++) {
      second_index = Math.round(12*Math.random());

      if (second_index != first_index) {
        break;
      }
    }

    // downloading the data
    let data_promise_1 = mnist.load(first_index);
    let data_promise_2 = mnist.load(second_index);
    // JS will not move past this line until both promises have resolved
    let [data1, data2] = await Promise.all([data_promise_1, data_promise_2]);

    // converting data to tensors of the right shape
    let imagesTensor1 = await tf.tensor2d(data1.images, [Math.floor(data1.images.length / 784), 784]);
    let labelsTensor1 = await tf.tensor2d(data1.labels, [Math.floor(data1.labels.length / 10), 10]);
    let imagesTensor2 = await tf.tensor2d(data2.images, [Math.floor(data2.images.length / 784), 784]);
    let labelsTensor2 = await tf.tensor2d(data2.labels, [Math.floor(data2.labels.length / 10), 10]);

    // concatenating them
    imagesTensor = tf.concat([imagesTensor1, imagesTensor2]);
    labelsTensor = tf.concat([labelsTensor1, labelsTensor2]);

    imagesTensor1.dispose();
    labelsTensor1.dispose();
    imagesTensor2.dispose();
    labelsTensor2.dispose();
  }

  await progress(0.2);

  // in worker, console.log sends the given string back to the client and so is async
  console.log('worker start');

  // the time that this worker will stop trianing and return to client
  const stop_time = shared_input.deploy_time + shared_input.run_time;

  // we now define our model
  const model = tf.sequential();

  model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
  model.add(tf.layers.dense({units: 20, activation: 'relu'}))
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

  model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

  // we now set the weights of our model
  let params = demarshall_parameters(shared_input.params);
  model.setWeights(params);

  const start_training_progress = 0.25;
  await progress(start_training_progress);

  console.log('starting training');

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
          progress((0.95 - start_training_progress)*((Date.now() - shared_input.deploy_time)/shared_input.run_time) + start_training_progress);
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

module.exports = {work: workFn}