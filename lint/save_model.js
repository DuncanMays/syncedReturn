const tf = require('@tensorflow/tfjs');
const lz = require('./lz_string.js');
const fs = require('fs');

function get_model() {
	const model = tf.sequential();

	model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
	model.add(tf.layers.dense({units: 20, activation: 'relu'}))
	model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

	model.compile({loss: tf.losses.softmaxCrossEntropy, metrics:[], optimizer: tf.train.adam()});

	return model;
}

// turns the parameter object that model.getWeights from an array of tensors into an array or arrays so that it is JSON serializable
function marshal_parameters(param_tensor) {
	let params = param_tensor.map(x => x.arraySync());
	return lz.compressToBase64(JSON.stringify(params));
}

const model = get_model();

const params = marshal_parameters(model.getWeights());

fs.writeFileSync('./out.txt', params)