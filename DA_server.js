const express = require('express')
const cors = require('cors');
const https = require('https');
const fs = require('fs');

const app = express()
const port = 8000

const data_allocation_array = JSON.parse(JSON.parse(fs.readFileSync('data_allocation.txt').toString()));
const data_allocation = {};

for (let i=0; i<data_allocation_array.length; i++) {
	a = data_allocation_array[i];
	data_allocation[a.ip_addr] = a
}

app.use(cors());

app.get('/', (req, res) => {
	// getting the IP address of the requester
	const ip = req.headers['x-forwarded-for'] || req.connection.remoteAddress;

	console.log('got a request for data allocation from', ip)

	// the payload of the response
	let payload = JSON.stringify(data_allocation[ip].num_shards)

	if (payload == undefined){
		console.log('unrecognised IP');
	}

  res.send(payload);
});

https.createServer({
  key: fs.readFileSync('./sslcert/server.key'),
  cert: fs.readFileSync('./sslcert/server.cert')
}, app)
.listen(port, function () {
  console.log('Example app listening on port', port)
})