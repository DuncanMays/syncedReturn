// needed to collect command line arguements
const process = require('process');

// this is the function that will be sent to each worker
async function worker(input) {
  // we must call progress every 30 seconds or the scheduler will think the worker has died

  let { lazy_load } = require('lazy_loader');
  await lazy_load(['train_shard_1']);
  mnist_shard = require('train_shard_1');

  const lz = require('lzstring');

  data = JSON.parse(lz.decompressFromBase64(mnist_shard));

  progress(1);
  return data.labels;
}

async function main() {
  // the compute api
  await require('dcp-client').init(process.argv);
  const compute = require('dcp/compute');
  
  // this is a critical line, calling compute.for creates a job, which is the central datastructure of the compute API
  // the first parameter is a list of inputs for each worker. Each worker will get one element of this list and execute the worker function on it. So this job will recruit 3 workers, which will get 1, 2, and 3 as inputs. This list must be JSON serializable
  // the second parameter is a function that each worker will call on their given input. This function must also be JSON serializable and not reference any local variables like other loaded modules or node-specific functions.
  let job = compute.for([1,2,3], worker);

  // jobs are event emmiters and can have different handlers attached to them
  // this event happens when the job is accepted by the scheduler, at which point we log "Job accepted was accepted by the scheduler"
  job.on('accepted', () => {
    console.log("Job accepted was accepted by the scheduler");
  });

  // a status event is emmited whenever a status update is given by the scheduler.
  job.on('status', (status) => {
    console.log("Got a status update:", status);
  });

  job.on('error', (err) => {console.log('there was an error: ', err);});

  // the slices (the subset of the job characterized by a single input element and work function) each go to different workers, and so will complete and send their results at different times. This event is emmited when one worker sends their result back.
  job.on('result', (value) => console.log("Got a result:", value.result));

  job.requires('mnist_shards_500/train_shard_1');
  job.requires('lzstring/lzstring');
  job.requires('lazy_loader/lazy_loader');

  // this call actually contacts the scheduler and tells it about this job. It's parameter is the amount of DCC that you are bidding on this job. compute.marketValue is simply the market value of a typical slice on the network, based on statistics accessible to the scheduler.
  let results = await job.exec(0.01);
  // console.log(100*compute.marketValue);

  // some events that jobs emit don't occur before the completion of every job. This means that event handlers attached to jobs that have completed can listen perminintly without ever running, and prevent node from shutting down. For this reason it's good practice to call process.exit() at the end of the program
  process.exit()
}

main()
