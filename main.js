
wrk = require('./work_function_1.js');

worker_config = {
  benchmark_length: 500,
  time_for_training: 60
};

async function main() {

  await wrk(0, worker_config); 

}

main()