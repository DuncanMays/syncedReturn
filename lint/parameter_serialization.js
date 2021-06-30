const tf = require('@tensorflow/tfjs');
const lz = require('./lz_string.js');

/// turns the parameter object that model.getWeights from an array of tensors into an array or arrays so that it is JSON serializable
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

function get_model() {
  const model = tf.sequential();

  model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
  model.add(tf.layers.dense({units: 20, activation: 'relu'}))
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

  model.compile({loss: tf.losses.softmaxCrossEntropy, metrics:['accuracy'], optimizer: tf.train.adam()});

  return model;
}

model1 = get_model();
model2 = get_model();

marshalled_params = marshal_parameters(model1.getWeights());

console.log(marshalled_params.length);

new_params = demarshall_parameters(marshalled_params);

model2.setWeights(new_params);
