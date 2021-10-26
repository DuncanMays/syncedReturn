import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('./data/timeVaccuracy.csv')

work_function_1 = (data['work function 1'] + data['Unnamed: 2'] + data['Unnamed: 3']).to_numpy()/3
work_function_2 = (data['work function 2'] + data['Unnamed: 5'] + data['Unnamed: 6']).to_numpy()/3
work_function_3 = (data['work function 3'] + data['Unnamed: 8'] + data['Unnamed: 9']).to_numpy()/3

# filtering out null values at the beginning

work_function_1 = work_function_1[5:]
work_function_2 = work_function_2[5:]
work_function_3 = work_function_3[5:]

x = list(range(work_function_1.shape[0]))

plt.plot(x, work_function_1)
plt.plot(x, work_function_2)
plt.plot(x, work_function_3)

plt.show()