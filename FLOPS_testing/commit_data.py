# this script reads data from head.data and appends it all to the data recorded in commit.data

import json

f = open('./head.data', 'r')
head = json.loads(f.read())
f.close()

f = open('./commit.data', 'r')
commit = json.loads(f.read())
f.close()

appended = commit + head

f = open('./commit.data', 'w')
f.write(json.dumps(appended))
f.close()