// needed to collect command line arguements
const process = require('process');
const fs = require('fs');

const work_fn = require('./init_work_function.js');

async function main() {
  // the compute api
  await require('dcp-client').init(process.argv);
  const compute = require('dcp/compute');
  const { RemoteDataSet } = require('dcp/compute');

  // const slice_URL = new URL('https://localhost:7999');
  const slice_URL = new URL('https://192.168.2.19:7999');
  const URL_list = [];
  for (let i=0; i<5; i++){
    URL_list.push(slice_URL)
  }

  let remoteDataSet = new RemoteDataSet(URL_list)

  let job = compute.for(remoteDataSet, work_fn);
  // let job = compute.for(URL_list, work_fn);

  job.on('accepted', () => {console.log("Job accepted was accepted by the scheduler, id:", job.address.slice(0, 15));});
  job.on('status', (status) => {console.log("Got a status update:", status);});
  job.on('result', (value) => {console.log("Got a result:", value);});
  job.on('error', (err) => {console.log('there was an error: ', err);});
  job.on('console', (msg) => {console.log("worker "+msg.sliceIndex+" logged: "+msg.message);});

  job.requires('aistensorflow/tfjs');
  job.requires('mnist_shards_500/train_shard_1');
  job.requires('lzstring/lzstring');
  job.requires('lazy_loader/lazy_loader');

  job.computeGroups = [{joinKey: 'queens-edge', joinSecret: 'P8PuQ0oCXm'}]

  // this call actually contacts the scheduler and tells it about this job. It's parameter is the amount of DCC that you are bidding on this job. compute.marketValue is simply the market value of a typical slice on the network, based on statistics accessible to the scheduler.
  let results = await job.exec(0.001);
  // console.log(100*compute.marketValue);

  fs.writeFileSync('data_allocation.txt', JSON.stringify(results));

  // some events that jobs emit don't occur before the completion of every job. This means that event handlers attached to jobs that have completed can listen perminintly without ever running, and prevent node from shutting down. For this reason it's good practice to call process.exit() at the end of the program
  process.exit()
}

main()