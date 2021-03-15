const tf = require("@tensorflow/tfjs-node");
const mnist = require('mnist');

// define a model

const model = tf.sequential();

model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
model.add(tf.layers.dense({units: 20, activation: 'relu'}))
model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

model.compile({loss: tf.losses.meanSquaredError, metrics:['accuracy'], optimizer: tf.train.adam()});

// we now import the MNIST dataset

const num_training = 8000
const num_testing = 2000

var set = mnist.set(num_training, num_testing);

var trainingSet = set.training;
var testingSet = set.test;

// we now format the data as tensors of the appropriate shape

let x_train = [];
let y_train = [];
let x_test = [];
let y_test = [];

trainingSet.map(sample => x_train.push(sample.input));
trainingSet.map(sample => y_train.push(sample.output));
testingSet.map(sample => x_test.push(sample.input));
testingSet.map(sample => y_test.push(sample.output));

x_train = tf.tensor(x_train);
y_train = tf.tensor(y_train);
x_test = tf.tensor(x_test);
y_test = tf.tensor(y_test);

// sets training data to random noise, this is to test how much sparsity affects things
// x_train = tf.randomNormal(x_train.shape);
// y_train = tf.randomNormal(y_train.shape);

// training model on data, while keeping track of the number of the amount of time each epoch took to give a samples per second calculation

let epoch_start = 0;
let epoch_end = 0;
let time_for_epoch, samples_per_second;

epoch_start = new Date().getTime();

model.fit(x_train, y_train, {
  epochs: 10,
  batch_size: 32,
  callbacks: {
    onEpochEnd: (batch, logs) => {
      epoch_end = new Date().getTime();

      time_for_epoch = epoch_end - epoch_start;
      samples_per_second = 1000*num_training/time_for_epoch;
      console.log('samples per second during epoch: '+samples_per_second);

      epoch_start = new Date().getTime();
    }
  }
});