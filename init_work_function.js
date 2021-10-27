
// this is the function that will be sent to each worker
module.exports = async function worker(slice_input) {
	// imports the required modules
	tf = require('tfjs');

	const { lazy_load } = require('lazy_loader');
	const lz = require('lzstring');

	// these lines might make workers crash
	tf.setBackend('cpu');
	await tf.ready();

	// returns the MNIST model that we will train here
	function get_model() {
		const model = tf.sequential();

		model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
		model.add(tf.layers.dense({units: 20, activation: 'relu'}))
		model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

		model.compile({loss: tf.losses.softmaxCrossEntropy, metrics:[], optimizer: tf.train.adam()});

		return model;
	}

	function FLOPS_benchmark(){
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

		let FLOPS = iters*ops/elapsed;

		return FLOPS
	}

	// 500 samples per data shard
	const NUM_DUMMY_SAMPLES = 500;
	async function subset_benchmark(){
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
					epochs_per_second = 1000/time_for_epoch;

					// pushes the training rate to a list
					rate_data.push(epochs_per_second)

					// this callback runs in between epochs, and so the end of this function is the start of the next epoch
					epoch_start = Date.now();
				}// onEpochEnd
			}// callbacks
		});// .fit

		// we will now take the average of the epochs_per_second of the epochs
		total = 0
		for (let i=0; i<rate_data.length; i++) {
			total = total + rate_data[i];
		}
		const avg_rate = total/num_epochs;

		return avg_rate
	}

	async function communication_benchmark(){

		const start = Date.now();

		// downloads and decompresses data
		await lazy_load(['train_shard_1']);
		mnist_shard = require('train_shard_1');

		raw = JSON.parse(lz.decompressFromBase64(mnist_shard));

		const end = Date.now();

		// 1000 because Data.now() reports time in milliseconds
		const rate = 1000/(end - start);

		return rate;
	}

	// we must call progress every 30 seconds or the scheduler will think the worker has died
	progress(0.1);
	
	const comp = await subset_benchmark();

	progress(0.5);

	const comm = await communication_benchmark();

	progress(0.6);

	const flops = FLOPS_benchmark();

	progress(0.9);

	// the training deadline, in seconds
	const D = 60;
	// P_d is the amount of data in each data shard, which is one since a data shard was the measurement unit
	const P_d = 1;
	// P_m is the amount of data in the model's parameters, which must be expressed as a ratio wrt the size of a data shard
	const P_m = 0.354;
	// mu is the number of training iterations, we should probably set this automatically, but for the time being this is fine
	const mu = 3;

	let num_shards = (D - 2*P_m/comm)/(mu/comp + P_d/comm)

	num_shards = Math.floor(num_shards);
	num_shards = Math.min(num_shards, 120);

	const return_obj = {
		'ip_addr': slice_input.ip_addr,
		'num_shards': num_shards,
		'FLOPS_score': flops,
		'subset_score': comp,
		'download_score':comm
	};

	progress(1);

	return return_obj;
}