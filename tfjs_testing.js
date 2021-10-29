const mnist = require('mnist');
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const process = require('process');

// returns the MNIST model that we will train here
function get_model() {
  const model = tf.sequential();

  model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
  model.add(tf.layers.dense({units: 20, activation: 'relu'}))
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

  model.compile({loss: tf.losses.softmaxCrossEntropy, metrics:['accuracy'], optimizer: tf.train.adam()});

  return model;
}

console.log('loading data');
const dataset = JSON.parse(fs.readFileSync('./preprocessedMNIST.txt'))
console.log('done loading data');

const train_x_raw = dataset['train_x']
const train_y_raw = dataset['train_y']

train_x = tf.tensor(train_x_raw).slice(0, 10000);
train_y = tf.tensor(train_y_raw).slice(0, 10000);

const model = get_model();

async function main(){
  console.log('training');
  model.fit(train_x, train_y, {
  	batchSize: train_x.shape[0],
  	epochs: 1,
  	verbose: 1,
    yieldEvery: 1000,
    callbacks: {
      onYield: (epoch, batch, logs) => {
        console.log('yeild')
      }
    }
  });
}

setInterval(() => {console.log('ping');}, 1000);

main();
