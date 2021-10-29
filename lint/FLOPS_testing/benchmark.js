const tf = require('@tensorflow/tfjs-node');

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

NUM_DUMMY_SAMPLES = 500
async function get_sample_rate(){
	const dummy_model = tf.sequential();

    dummy_model.add(tf.layers.dense({units: 50, inputShape: [784], activation: 'relu'}))
    dummy_model.add(tf.layers.dense({units: 20, activation: 'relu'}))
    dummy_model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

    dummy_model.compile({loss: tf.losses.meanSquaredError, metrics:[], optimizer: tf.train.adam()});

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
			samples_per_second = 1000*x_dummy.shape[0]/time_for_epoch;

			// pushes the training rate to a list
			rate_data.push(samples_per_second)

			// this callback runs in between epochs, and so the end of this function is the start of the next epoch
			epoch_start = Date.now();
		}
	}
	});

	// we will now take the average of the samples_per_second of the epochs
	total = 0
	for (let i=0; i<rate_data.length; i++) {
	total = total + rate_data[i];
	}
	sample_rate = total/num_epochs;

	return sample_rate
}

async function main(){
	// console.log(benchmark());
	console.log(await get_sample_rate());
}

main()
