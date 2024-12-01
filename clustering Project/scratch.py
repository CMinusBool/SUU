import pandas as pd
import numpy as np

print('Month')
for i in range(1, 13):
    print(i)
    print(np.cos(2 * np.pi * i / 12))
    print('---')

print('Day')
for i in range(1, 32):
    print(i)
    print(np.cos(2 * np.pi * i / 31))
    print('---')
#b = df_encoded['Commissioned_day_cos'] = np.cos(2 * np.pi * df_encoded['Commissioned_day'] / 31)