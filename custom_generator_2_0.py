# Custom data generator for train GTSM surrogate model
# REF:     [ https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-tfk-self.LSTM_steps_in52b64e41c3 ]
# EXAMPLE: [https://github.com/PimpMyGit/BriGan/blob/master/main.ipynb]

import tensorflow as tf
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
import os        

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, lat, lon, lat_range,lon_range,nearest_station_index,train_y_bounds,train_month_start_end, shuffle, LSTM_steps_in,ms_recurrent_steps, ERA5_var,
                 batch_size,fillnan,output_type):
        
        self.df = df.copy() # pandas dataframe with paths
        self.lat = lat # GTSM station lat
        self.lon = lon # GTSM station lon
        self.lat_range = lat_range # lat min/max
        self.lon_range = lon_range # lon min/max
        self.nearest_station_index = nearest_station_index # station index for GTSM dataset
        self.train_y_bounds = train_y_bounds # (y_start, y_stop)
        self.train_month_start_end = train_month_start_end # (m_start, m_stop)
        self.batch_size = batch_size # numer of .grib files to generate training batches (i.e. not real batch size)
        self.LSTM_steps_in = LSTM_steps_in # steps for each ERA5 input in LSTM
        self.ms_recurrent_steps = ms_recurrent_steps # steps for az and alt input in LSTM
        self.shuffle = bool(shuffle) # shuffle datset at each epoch True of False
        self.ERA5_var = ERA5_var # variables in ERA5 datasets
        self.fillnan = fillnan # False or true for fill nan with rolling window (3 steps window)
        self.output_type = output_type # True  = type 2 (surge,tide_only) else (surge+tide)
        # self.n = sum([self.__get_len(filepath) for filepath in df["ERA5_path"]]) # number of inputs for the NN train (must be adjusted by past vect. size)
        self.n = len(df["ERA5_path"]) # number of elements in files-path dataframe

    # def __get_len(self, paths):
    #     a = xr.open_dataset(str(paths), engine='cfgrib')
    #     print(paths)
    #     len_grib = len(a[self.ERA5_var[0]].sel(latitude=self.lat, longitude=self.lon, method="nearest").values)
    #     del a
    #     return len_grib

    # Create mooving windows with stride for ms
    def create_sliding_windows(self,data, window_size):
        shape = (data.shape[0] - window_size + 1, window_size, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    
    # Create mooving windows with stride for ERA5
    def rolling_window(self,a, window):
            shape = (a.shape[0], a.shape[1] - window + 1, window, a.shape[2], a.shape[3])
            strides = (a.strides[0], a.strides[1], a.strides[1], a.strides[2], a.strides[3])
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def __get_input(self, in_paths,in_paths_2,in_paths_3):  
        #_______PREV____PATH______________________________________________

        current_path = in_paths
        current_path_2 = in_paths_2
        current_path_3 = in_paths_3

        basename = os.path.basename(current_path)
        year_month = basename.split('_')[-1].split('.')[0]  # '197903'
        year = int(year_month[:4])
        month = int(year_month[4:])

        if month == 1:
            prev_year = year - 1
            prev_month = 12
        else:
            prev_year = year
            prev_month = month - 1

        prev_month_str = f'{prev_month:02}'

        previous_path = current_path.replace(f'{year_month}', f'{prev_year}{prev_month_str}')
        previous_path_2 = current_path_2.replace(f'{year_month}', f'{prev_year}{prev_month_str}')
        previous_path_3 = current_path_3.replace(f'{year_month}', f'{prev_year}{prev_month_str}')
        #_______________________________________________________________
        #_______LOAD____USEFULL____FILES________________________________
        a = xr.open_dataset(str(in_paths), engine='cfgrib')
        previous_dataset = xr.open_dataset(previous_path, engine='cfgrib')
        # slice bbox
        b=a[self.ERA5_var].sel(latitude=slice(*self.lat_range), longitude=slice(*self.lon_range))
        selected_previous = previous_dataset[self.ERA5_var].sel(latitude=slice(*self.lat_range), longitude=slice(*self.lon_range))

        last_previous = selected_previous.isel(time=slice(-(self.LSTM_steps_in-1), None))

        combined_dataset = xr.concat([last_previous, b], dim='time')
        print("In: Initial ERA5 shape"+str(combined_dataset.dims))
        a.close()
        previous_dataset.close()
        # load altitude and azimute
        sm = pd.read_csv(in_paths_2, delimiter=' ')
        sm_2 = pd.read_csv(previous_path_2, delimiter=' ')
        sm = pd.concat([sm_2.tail(self.LSTM_steps_in-1), sm], ignore_index=True)
        print("In: Initial ms shape"+str(sm.shape))
        del sm_2
        # load mean sea level
        mean_sl = pd.read_csv(in_paths_3, delimiter=' ')
        mean_sl_2 = pd.read_csv(previous_path_3, delimiter=' ')
        mean_sl = pd.concat([mean_sl_2.tail(self.LSTM_steps_in-1), mean_sl], ignore_index=True)
        print("In: Initial mean_sl shape"+str(mean_sl.shape))
        #_______________________________________________________________
        #_______GET____LAT____LON_______________________________________

        latitudes = combined_dataset['latitude'].values
        longitudes = combined_dataset['longitude'].values

        latlon = np.zeros((sm.shape[0]-(self.LSTM_steps_in-1), 2, latitudes.shape[0], longitudes.shape[0]))

        for i in range(latitudes.shape[0]):
            for j in range(longitudes.shape[0]):
                latlon[:, 0, i, j] = latitudes[i]
                latlon[:, 1, i, j] = longitudes[j]
        print("In: lat_lon layer shape"+str(latlon.shape))
        #_______________________________________________________________
        #_______CREATE____.GRIB____SUB-BATCHES__________________________

        combined_dataset=combined_dataset.to_array()

        combined_dataset = combined_dataset.values

        windows = self.rolling_window(combined_dataset, self.LSTM_steps_in)
        print("In: final ERA5 shape"+str(windows.shape))

        reshaped_numpy_dataset = np.zeros((windows.shape[1], windows.shape[0] * windows.shape[2], windows.shape[3], windows.shape[4]))

        for j in range( windows.shape[0] ):
                reshaped_numpy_dataset[:, j*windows.shape[2]:(j+1)*windows.shape[2], :, :] = windows[j, :, :, :, :]
        print("In: Final ERA5 reshape"+str(reshaped_numpy_dataset.shape))
        #_______________________________________________________________
        #_______CREATE____MS.TXT____SUB-BATCHES_________________________
        sm = sm[['sun_azimut', 'sun_altitude', 'moon_azimut', 'moon_altitude']].values
        # Create mooving window for recurrent steps
        sliding_windows = self.create_sliding_windows(sm, self.ms_recurrent_steps)

        sliding_windows = sliding_windows[:, :, :, np.newaxis, np.newaxis]

        print("In: slide ms shape"+str(sliding_windows.shape))
        sliding_windows = sliding_windows.transpose(0, 2, 1, 3, 4)
        print("In: slide ms reshape"+str(sliding_windows.shape))
        final_shape = sliding_windows.shape

        reshaped_numpy_dataset_2 = np.zeros((final_shape[0], final_shape[1] * final_shape[2], final_shape[3], final_shape[4]))

        for j in range(final_shape[1]):
                reshaped_numpy_dataset_2[:, j*final_shape[2]:(j+1)*final_shape[2], :, :] = sliding_windows[:, j, :, :, :]
        print("In: Final ms reshape"+str(reshaped_numpy_dataset_2.shape))
        reshaped_numpy_dataset_2 = np.repeat(reshaped_numpy_dataset_2, reshaped_numpy_dataset.shape[2], axis=2)
        reshaped_numpy_dataset_2 = np.repeat(reshaped_numpy_dataset_2, reshaped_numpy_dataset.shape[3], axis=3)
        print("In: Final ms adjusted"+str(reshaped_numpy_dataset_2[(self.LSTM_steps_in-self.ms_recurrent_steps):,:,:,:].shape))
        #_______________________________________________________________
        #_______CREATE____MEAN_SL.TXT____SUB-BATCHES____________________
        mean_sl = mean_sl[['msl']].values
        mean_sl = np.expand_dims(mean_sl, axis=-1) 
        mean_sl = np.expand_dims(mean_sl, axis=-1) 
        mean_sl = np.repeat(mean_sl, reshaped_numpy_dataset.shape[2], axis=2)
        mean_sl = np.repeat(mean_sl, reshaped_numpy_dataset.shape[3], axis=3)
        print("In: Final mean_sl shape"+str(mean_sl[(self.LSTM_steps_in-1):,:,:,:].shape))
        #_______________________________________________________________
        #_______CONCATENATE____DATASETS_________________________________
        reshaped_numpy_dataset = np.concatenate((reshaped_numpy_dataset, reshaped_numpy_dataset_2[(self.LSTM_steps_in-self.ms_recurrent_steps):,:,:,:],mean_sl[(self.LSTM_steps_in-1):,:,:,:],latlon), axis=1)
        #_______________________________________________________________
        # return inputs (sub-batches) from each .grib
        return reshaped_numpy_dataset
    
    def __get_output(self, out_paths, out_paths_2, out_paths_3):
        
        #_______PREV____PATH______________________________________________

        current_path_3 = out_paths_3

        basename = os.path.basename(current_path_3)
        year_month = basename.split('_')[-1].split('.')[0]  # '197903'
        year = int(year_month[:4])
        month = int(year_month[4:])      
        #_______________________________________________________________
        #_______LOAD____USEFULL____FILES________________________________
        wl = xr.open_dataset(out_paths_2)
        wl = wl["waterlevel"].isel(stations=self.nearest_station_index).values[:,None] 
        print("Out: Initial wl shape"+str(wl.shape))
        # load mean sea level
        mean_sea_level = pd.read_csv(out_paths_3, delimiter=' ')
        mean_sea_level = mean_sea_level[['msl']].values
        print("Out: Initial msl shape"+str(mean_sea_level.shape))
        #_______________________________________________________________
        #_______DEFINE____OUTPUT________________________________________       
        tide_plus_surge = wl-mean_sea_level
        if (year==self.train_y_bounds[0] and month == self.train_month_start_end[0]): # if it is the first file reshape
             tide_plus_surge = tide_plus_surge[self.LSTM_steps_in-1:,:]

        if (np.any(np.isnan(tide_plus_surge)) and self.fillnan):
            data_series = pd.Series(tide_plus_surge[:,0])
            tide_plus_surge[:,0] = data_series.fillna(data_series.rolling(window=3, min_periods=1).mean()).to_numpy()

        print("Out:  tide_plus_surge adjusted shape"+str(tide_plus_surge.shape))
        return tide_plus_surge
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_arrays = [self.__get_input(ERA5_paths,solar_lunar_path,mean_sea_level_path) for ERA5_paths,solar_lunar_path,mean_sea_level_path  in zip(batches["ERA5_path"], batches["solar_lunar_path"], batches["mean_sea_level"])]
        X_batch = np.asarray(np.concatenate(X_arrays, axis=0))

        if self.output_type:
            y_arrays = [self.__get_output_2(surge_path,waterlevel_path,mean_sea_level_path) for surge_path,waterlevel_path,mean_sea_level_path  in zip(batches["surge"], batches["waterlevel"], batches["mean_sea_level"])]
            y_batch = np.asarray(np.concatenate(y_arrays, axis=0))
        else:
            y_arrays = [self.__get_output(surge_path,waterlevel_path,mean_sea_level_path) for surge_path,waterlevel_path,mean_sea_level_path  in zip(batches["surge"], batches["waterlevel"], batches["mean_sea_level"])]
            y_batch = np.asarray(np.concatenate(y_arrays, axis=0))

        return X_batch, y_batch
    
    def __getitem__(self, index):
        batches = self.df[index*self.batch_size : (index+1)*self.batch_size]
        X, y = self.__get_data(batches)      
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return self.n // self.batch_size
    

    def __get_output_2(self, out_paths, out_paths_2, out_paths_3):
        
        #___________PATH_________________________________________________
        current_path_3 = out_paths_3
        basename = os.path.basename(current_path_3)
        year_month = basename.split('_')[-1].split('.')[0]  # '197903'
        year = int(year_month[:4])
        month = int(year_month[4:])      
        #_______________________________________________________________
        #_______LOAD____USEFULL____FILES________________________________
        # surge
        surge = xr.open_dataset(out_paths)
        surge = surge["surge"].isel(stations=self.nearest_station_index).values[:,None] 
        print("Out: Initial surge shape"+str(surge.shape))
        # waterlevel
        wl = xr.open_dataset(out_paths_2)
        wl = wl["waterlevel"].isel(stations=self.nearest_station_index).values[:,None] 
        print("Out: Initial wl shape"+str(wl.shape))
        # load mean sea level
        mean_sea_level = pd.read_csv(out_paths_3, delimiter=' ')
        mean_sea_level = mean_sea_level[['msl']].values
        print("Out: Initial msl shape"+str(mean_sea_level.shape))
        #_______________________________________________________________
        #_______DEFINE____OUTPUT________________________________________       

        if (np.any(np.isnan(wl)) and self.fillnan):
            data_series = pd.Series(wl[:,0])
            wl[:,0] = data_series.fillna(data_series.rolling(window=3, min_periods=1).mean()).to_numpy()
        if (np.any(np.isnan(mean_sea_level)) and self.fillnan):
            data_series = pd.Series(mean_sea_level[:,0])
            mean_sea_level[:,0] = data_series.fillna(data_series.rolling(window=3, min_periods=1).mean()).to_numpy()
        if (np.any(np.isnan(surge)) and self.fillnan):
            data_series = pd.Series(surge[:,0])
            surge[:,0] = data_series.fillna(data_series.rolling(window=3, min_periods=1).mean()).to_numpy()

        tide = wl-mean_sea_level-surge
        tide_and_surge = np.ones([tide.shape[0],2])
        tide_and_surge[:,0:1] = tide
        tide_and_surge[:,1:2] = surge
        print("Out:  surge adjusted shape"+str(tide.shape))
        print("Out:  surge adjusted shape"+str(surge.shape))
        return tide_and_surge