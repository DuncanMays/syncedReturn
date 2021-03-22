import json
from matplotlib import pyplot as plt

f = open('./results.data', 'r')
raw = f.read()
f.close()

results = json.loads(raw)

x = [s['benchmark_score'] for s in results]
y = [s['sample_rate'] for s in results]

plt.scatter(x, y)
plt.xlabel('benchmark score')
plt.ylabel('training rate (samples per second)')
plt.show()