const process = require('process');

async function worker(slice_input, shared_input) {
  progress(1);
  // return shared_input;
  return 'shared input was '+shared_input.length+' characters long';
}

// the compressed model parameters are 473681 characters long

// creates input string of specified length
const input_length = 473681;
let input_string = '';
for (let i=0; i<input_length; i++) {
  input_string = input_string.concat('a');
}

async function main() {
  await require('dcp-client').init(process.argv);
  const compute = require('dcp/compute');

  const slices = [];

  for (let i=0; i<5; i++) {
    slices.push('slice input');
  }
  
  let job = compute.for(slices, worker, [input_string]);

  job.on('accepted', () => {
    console.log("Job accepted was accepted by the scheduler");
  });

  job.on('status', (status) => {
    console.log("Got a status update:", status);
  });

  job.on('result', (value) => {
    console.log("Got a result:", value);
  });

  let results = await job.exec(0.01);

  console.log('input length: ', input_length);

  process.exit()
}

main();
