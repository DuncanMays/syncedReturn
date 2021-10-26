const express = require('express')
const cors = require('cors');
const https = require('https');
const fs = require('fs');

const app = express()
const port = 7999

app.use(cors());

app.get('/', (req, res) => {
	// getting the IP address of the requester
	const ip = req.headers['x-forwarded-for'] || req.connection.remoteAddress;

	console.log('got a GET request from', ip)

	// the payload of the response
	let payload = {'ip_addr':ip};

  res.send(payload);
});

https.createServer({
  key: fs.readFileSync('./sslcert/server.key'),
  cert: fs.readFileSync('./sslcert/server.cert')
}, app)
.listen(port, function () {
  console.log('Example app listening on port', port)
})