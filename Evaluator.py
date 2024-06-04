import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import datetime
import pytz
# import tensorflow as tf
# from keras import Sequential
import keras
import os.path
from os import path
from sklearn.model_selection import train_test_split



# for real evaluation
# start_date = datetime.date(2024, 5, 1)
# end_date = datetime.date(2024, 5, 31)

# for test evaluation
start_date = datetime.date(2024, 4, 1)
end_date = datetime.date(2024, 4, 30)

OUTLIER_MAX = 10000000
OUTLIER_MIN = 0

LOCATIONS = [
    'valor00001', 'valor00002', 'yaoko00006', 'yaoko00007'
]




def rmr_score(actual, predict):
    # Drop outlier value
    merged_df = pd.concat(
        [pd.DataFrame(actual, columns=['actual']), pd.DataFrame(predict, columns=['predict'])],
        axis=1
    )
    filtered_df = merged_df.loc[
        (merged_df['actual'] >= OUTLIER_MIN) & (merged_df['actual'] <= OUTLIER_MAX)
    ]
    actual = filtered_df['actual']
    predict = filtered_df['predict']

    eps = 1e-9
    actual = actual + eps
    diff = actual - predict
    mx = sum(abs(diff)) / sum(actual)
    
    return mx * 100

def merge_data(files,dir_path,location):
    main = [files[i] for i in range(0, 5) if "demand" in files[i]][0]
    temperature = [files[i] for i in range(0, 5) if "temperature(℃)" in files[i]][0]
    solar = [files[i] for i in range(0, 5) if "solar(W/m2)" in files[i]][0]
    telop = [files[i] for i in range(0, 5) if "telop" in files[i]][0]
    cloud = [files[i] for i in range(0, 5) if "cloud(%)" in files[i]][0]


    UTC = "+09:00"
    main["datetime"] = main["time"].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).isoformat()+UTC)

    # merge['datetime'] = pd.to_datetime(merge['datetime'])
    # main['datetime'] = pd.to_datetime(main['datetime'])
    full_data = pd.merge(main, cloud[['datetime', 'cloud(%)']], on='datetime', how='left')
    full_data = pd.merge(full_data, solar[['datetime', 'solar(W/m2)']], on='datetime', how='left')
    full_data = pd.merge(full_data, telop[['datetime', 'telop']], on='datetime', how='left')
    full_data = pd.merge(full_data, temperature[['datetime', 'temperature(℃)']], on='datetime', how='left')

    # Fill missing values for weather-related features with their mean values
    full_data['cloud(%)'].fillna(full_data['cloud(%)'].mean(), inplace=True)
    full_data['solar(W/m2)'].fillna(full_data['solar(W/m2)'].mean(), inplace=True)
    full_data['temperature(℃)'].fillna(full_data['temperature(℃)'].mean(), inplace=True)
    full_data['telop'].fillna(full_data['telop'].mode()[0], inplace=True) 
    full_data['demand'].fillna(full_data['demand'].mean(),inplace=True)
    full_data['solar'].fillna(full_data['solar'].mean(),inplace=True)

    return full_data

def train_model(index):
    location = LOCATIONS[index]
    dir_path = Path(__file__).parent / 'share_participants'
    input_files = list(dir_path.glob(f"{location}*.csv"))
    files = []
    for input_file in input_files:
        files.append(pd.read_csv(input_file))
    assert len(files) == 5
    full_data = merge_data(files,dir_path,location)


    X = full_data[['cloud(%)', 'solar(W/m2)', 'telop', 'temperature(℃)']].to_numpy()
    y = full_data[['demand','solar']].to_numpy()


    # Split the data into training and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)  


    # Define the ANN model
    model = keras.Sequential()
    model.add(keras.layers.Dense(123,input_dim=4,activation="relu"))
    model.add(keras.layers.Dense(50,activation="relu"))
    model.add(keras.layers.Dense(8,activation="relu"))
    model.add(keras.layers.Dense(2))

    # Compile the model
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, Y_train, epochs=100, batch_size=8, validation_data=(X_val, Y_val))

    #save the model
    model.save(f"D:\sklearn-env\model\{location}.h5")

    #test
    y_pred = model.predict(X_test)
    print(y_pred)
    print(y_test)

def train():
    # train each files
    for i in range(0, len(LOCATIONS)):
        train_model(i)

def scan_data(location,file_data):
    name_files = ['_temperature_forcast','_solar_forcast','_telop_forcast','_cloud_forcast']
    datas = []
    # take tables have focast
    for file in name_files:
        # tail 48 => only take data in the target_day
        datas.append(file_data[location+file+".csv"].tail(48))
    temperature = datas[0]
    solar = datas[1]
    telop = datas[2]
    cloud = datas[3]
    #merge data
    full_data = pd.merge(solar[['datetime','solar(W/m2)']],telop[['datetime','telop']],on='datetime',how='inner')
    full_data = pd.merge(full_data,cloud[['datetime','cloud(%)']],on='datetime',how='inner')
    full_data = pd.merge(full_data,temperature[['datetime','temperature(℃)']],on='datetime',how='inner')
    
    return full_data[['cloud(%)', 'solar(W/m2)', 'telop', 'temperature(℃)']].to_numpy()

def predict(location, target_date, input_feature_dfs):
    dir_path = Path(__file__).parent/'model'
    cond = path.exists(f"{dir_path}\\{location}.h5") 

    # if model hasnt been trained, train model
    if not cond:
        train()
        predict(location, target_date, input_feature_dfs)
    
    data = scan_data(location,input_feature_dfs)
    #define model
    model = keras.Sequential()
    model.add(keras.layers.Dense(123,input_dim=4,activation="relu"))
    model.add(keras.layers.Dense(50,activation="relu"))
    model.add(keras.layers.Dense(8,activation="relu"))
    model.add(keras.layers.Dense(2))
    # print(f"{dir_path}\\{location}.h5")

    #load model exist
    model.load_weights(f"{dir_path}\\{location}.h5")
    data_pred = model.predict(data)
    
    power_demand,power_generation = data_pred[:,0],data_pred[:,1]
    print(power_demand)
    return power_generation, power_demand


def evaluate():
    metric_generations = []
    metric_demands = []

    dir_path = Path(__file__).parent / 'share_participants'

    # evaluate for each location
    for location in LOCATIONS:
        input_files = list(dir_path.glob(f"{location}*.csv"))
        assert len(input_files) == 5

        original_dfs = {}
        for input_file in input_files:
            original_dfs[input_file.name] = pd.read_csv(input_file)

        target_date = start_date
        while target_date <= end_date:
            # make dataset for each target day
            input_feature_by = target_date - datetime.timedelta(days=2)
            print(f"predicting for location: {location}, date: {target_date.strftime('%Y-%m-%d')}")

            # filter df
            input_feature_dfs = {}
            for filename, df in original_dfs.items():
                input_feature_dfs[filename] = df[pd.to_datetime(df['time'], unit='ms') < pd.Timestamp(input_feature_by)]
                if 'forcast' in filename:
                    # check forecast data includes target day
                    assert max(pd.to_datetime(input_feature_dfs[filename]['datetime'])).date() == target_date

            # your prediction method
            power_generation_pred, power_demand_pred = predict(
                location, target_date, input_feature_dfs
            )

            # get truth data
            target_df = original_dfs[f"{location}.csv"]
            target_df = target_df[pd.to_datetime(target_df['time'], unit='ms').dt.date == pd.Timestamp(target_date).date()]

            assert len(target_df) == 48

            # get metric for 1 day
            metric_generation = rmr_score(actual=np.array(target_df['solar']), predict=np.array(power_generation_pred))
            metric_demand = rmr_score(actual=np.array(target_df['demand']), predict=np.array(power_demand_pred))

            metric_generations.append(metric_generation)
            metric_demands.append(metric_demand)

            # increment target_date
            target_date += datetime.timedelta(days=1)

    # get final metric (average for each day, each location)
    power_generation_metric = np.average(np.array(metric_generations))
    power_demand_truth_metric = np.average(np.array(metric_demands))
    return (power_generation_metric + power_demand_truth_metric) / 2


if __name__ == '__main__':
    metric = evaluate()
    print(f"your metrics is {metric}!")
