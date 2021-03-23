import json
from matplotlib import pyplot as plt

f = open('./commit.data', 'r')
raw = f.read()
f.close()

data = json.loads(raw)

x = [s['benchmark_score'] for s in data]
y = [s['sample_rate'] for s in data]

plt.scatter(x, y)
plt.xlabel('benchmark score')
plt.ylabel('training rate (samples per second)')
plt.show()