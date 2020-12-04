const tf = require('@tensorflow/tfjs-node');

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

console.log(rate);
