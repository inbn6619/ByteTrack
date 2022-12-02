import os
import pandas as pd

for num in range(1, 4):
    print(f'make_data {num} start')
    print(f'make_data {num} start')
    print(f'make_data {num} start')
    print(f'make_data {num} start')
    print(f'make_data {num} start')


    path = f'C:\\Users\\ai\\GangJuAI\\samples_vid\\samples_one_day\\data_samples\\{num}'

    data_names = os.listdir(path)

    paths = [f'{path}\\{name}' for name in data_names]

    df1 = pd.read_table(paths[0], sep=',')
    df2 = pd.read_table(paths[1], sep=',')
    df3 = pd.read_table(paths[2], sep=',')

    data_list = [df1, df2, df3]

    dataframe = pd.DataFrame()

    for i in range(max(df1['video_frame']) + 1):
        for data in data_list:
            print(f"num : {num}, idx : {i}")
            dataframe = pd.concat([dataframe, data[data['video_frame'] == i]])


    dataframe.to_csv(f"dataframe{num}.csv", mode='w')

    print(f'make_data {num} end')
    print(f'make_data {num} end')
    print(f'make_data {num} end')
    print(f'make_data {num} end')
    print(f'make_data {num} end')

