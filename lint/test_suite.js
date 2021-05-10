const tf = require("@tensorflow/tfjs");

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

function get_fake_result() {
    const model = tf.sequential();

    model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
    model.add(tf.layers.dense({units: 20, activation: 'relu'}))
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

    model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()}); 

    const return_obj = {
        completed_batches: Math.round(100*Math.random() + 100),
        params: marshal_parameters(model.getWeights())
    };

    return return_obj;
}

function get_fake_results(num_results) {
    let results = [];
    for (let i=0; i<num_results; i++) {
        results.push(get_fake_result());
    }
    return results
}

module.exports = {fake_results: get_fake_results};