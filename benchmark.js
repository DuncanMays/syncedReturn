const tf = require('@tensorflow/tfjs-node');

var matrix1 = tf.randomNormal([n,n], dtype='float32');
var matrix2 = tf.randomNormal([n,n], dtype='float32');
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

const rate = iters*ops/elapsed/1e6;