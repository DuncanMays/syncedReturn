# this script reads data from head.data and appends it all to the data recorded in commit.data

import json

def hash_data(data):
	a = 0
	# this hash fn should only operate on head files, of only around 50 data points, so a sequential operation like this shouldn't be terrible
	for d in data:
		a += int(d['sample_rate']*d['benchmark_score'])
	return a%1000000

f = open('./head.data', 'r')
head = json.loads(f.read())
f.close()

commit = None
try:
	f = open('./commit.data', 'r')
	commit = json.loads(f.read())
	f.close()
except(FileNotFoundError):
	commit = {'hash':0, 'data':[]}

head_hash = hash_data(head)
last_commit_hash = commit['hash']

if (head_hash == last_commit_hash):
	print('data already commited, not proceeding')
else:
	appended = commit['data'] + head

	to_write = {'hash':head_hash, 'data':appended}

	f = open('./commit.data', 'w')
	f.write(json.dumps(to_write))
	f.close()