import tensorflow as tf
import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
import numpy as np
import matplotlib.pyplot as plt

import custom_losses as cl

####################################################################################################################################

####################################################################################################################################
# Function for filter the list
def filter_files_highresmip(file_path, y_bounds, month_start_end):
    file_path_str = file_path.numpy().decode("utf-8")
    # get year and month for filtering criteria
    file_name = file_path_str.split('/')[-1]
    if file_name.endswith('.nc'):
        year_month = file_name.split('_')[-1].split('.')[0]
        year = int(year_month[:4])
        month = int(year_month[4:])

        if (year == y_bounds[0] and month_start_end[0] > month):
            return False
        if (year == y_bounds[1] and month_start_end[1] < month):
            return False
        if (y_bounds[0] <= year <= y_bounds[1]):
            return True   
    return False


# Function for filter the list
def filter_files(file_path, y_bounds, month_start_end):
    file_path_str = file_path.numpy().decode("utf-8")
    # get year and month for filtering criteria
    file_name = file_path_str.split('/')[-1]
    if file_name.endswith('.grib'):
        year_month = file_name.split('_')[-1].split('.')[0]
        year = int(year_month[:4])
        month = int(year_month[4:])

        if (year == y_bounds[0] and month_start_end[0] > month):
            return False
        if (year == y_bounds[1] and month_start_end[1] < month):
            return False
        if (y_bounds[0] <= year <= y_bounds[1]):
            return True

    if file_name.endswith('.txt'):
        year_month = file_name.split('_')[-1].split('.')[0]
        year = int(year_month[:4])
        month = int(year_month[4:])

        if (year == y_bounds[0] and month_start_end[0] > month):
            return False
        if (year == y_bounds[1] and month_start_end[1] < month):
            return False
        if (y_bounds[0] <= year <= y_bounds[1]):
            return True
        

    if file_name.endswith('.nc'):
        parts = file_name.split('_')
        year = int(parts[-3])
        month = int(parts[-2])
        
        if (year == y_bounds[0] and month_start_end[0] > month):
            return False
        if (year == y_bounds[1] and month_start_end[1] < month):
            return False
        if (y_bounds[0] <= year <= y_bounds[1]):
            return True
        
    return False

####################################################################################################################################

####################################################################################################################################

def filter_wrapper_highresmip(file_path, y_bounds , month_start_end):
    return tf.py_function(func=filter_files_highresmip, inp=[file_path, tf.constant(y_bounds), tf.constant(month_start_end)], Tout=tf.bool)

def filter_wrapper(file_path, y_bounds , month_start_end):
    return tf.py_function(func=filter_files, inp=[file_path, tf.constant(y_bounds), tf.constant(month_start_end)], Tout=tf.bool)

####################################################################################################################################

####################################################################################################################################


def compute_min_max(batch_X, y,LSTM_recurrent_steps,ms_recurrent_steps,output_type):
    
    if output_type:
        global_min = np.full((13), None)
        global_max = np.full((13), None)
        batch_min = np.full((13), None)
        batch_max = np.full((13), None)
    else:
        global_min = np.full((12), None)
        global_max = np.full((12), None)
        batch_min = np.full((12), None)
        batch_max = np.full((12), None)

    batch_min[0] = np.min(batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_min[1] = np.min(batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_min[2] = np.min(batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_min[3] = np.min(batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_min[4] = np.min(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_min[5] = np.min(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_min[6] = np.min(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_min[7] = np.min(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_min[8] = np.min(batch_X[:,-3,:,:]) 
    batch_min[9] = np.min(batch_X[:,-2,:,:]) 
    batch_min[10] = np.min(batch_X[:,-1,:,:]) 

    if output_type:
        batch_min[11] = np.min(y[:,0]) 
        batch_min[12] = np.min(y[:,1]) 

 
    else:
        batch_min[11] = np.min(y[:,:]) 


    batch_max[0] = np.max(batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_max[1] = np.max(batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_max[2] = np.max(batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_max[3] = np.max(batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_max[4] = np.max(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_max[5] = np.max(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_max[6] = np.max(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_max[7] = np.max(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_max[8] = np.max(batch_X[:,-3,:,:]) 
    batch_max[9] = np.max(batch_X[:,-2,:,:]) 
    batch_max[10] = np.max(batch_X[:,-1,:,:]) 

    if output_type:
        batch_max[11] = np.max(y[:,0]) 
        batch_max[12] = np.max(y[:,1]) 
    else:
        batch_max[11] = np.max(y[:,:]) 




    if output_type:
        for j in range(0,13):
            if global_min[j] is None:
                global_min[j] = batch_min[j]
            else:
                global_min[j] = np.minimum(global_min[j], batch_min[j])

            if global_max[j] is None:
                global_max[j] = batch_max[j]
            else:
                global_max[j] = np.maximum(global_max[j], batch_max[j])


    else:
        for j in range(0,12):
            if global_min[j] is None:
                global_min[j] = batch_min[j]
            else:
                global_min[j] = np.minimum(global_min[j], batch_min[j])

            if global_max[j] is None:
                global_max[j] = batch_max[j]
            else:
                global_max[j] = np.maximum(global_max[j], batch_max[j])

 
    return global_min, global_max

####################################################################################################################################

####################################################################################################################################


def compute_mean_std(batch_X, y,LSTM_recurrent_steps,ms_recurrent_steps,output_type):
    
    if output_type:
        global_mean = np.full((13), None)
        global_std = np.full((13), None)
        batch_mean = np.full((13), None)
        batch_std = np.full((13), None)
    else:
        global_mean = np.full((12), None)
        global_std = np.full((12), None)
        batch_mean = np.full((12), None)
        batch_std = np.full((12), None)

    batch_mean[0] = np.mean(batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_mean[1] = np.mean(batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_mean[2] = np.mean(batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_mean[3] = np.mean(batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_mean[4] = np.mean(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_mean[5] = np.mean(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_mean[6] = np.mean(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_mean[7] = np.mean(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_mean[8] = np.mean(batch_X[:,-3,:,:]) 
    batch_mean[9] = np.mean(batch_X[:,-2,:,:]) 
    batch_mean[10] = np.mean(batch_X[:,-1,:,:]) 

    if output_type:
        batch_mean[11] = np.mean(y[:,0]) 
        batch_mean[12] = np.mean(y[:,1]) 

 
    else:
        batch_mean[11] = np.mean(y[:,:]) 


    batch_std[0] = np.std(batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_std[1] = np.std(batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_std[2] = np.std(batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_std[3] = np.std(batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_std[4] = np.std(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_std[5] = np.std(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_std[6] = np.std(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_std[7] = np.std(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_std[8] = np.std(batch_X[:,-3,:,:]) 
    batch_std[9] = np.std(batch_X[:,-2,:,:]) 
    batch_std[10] = np.std(batch_X[:,-1,:,:]) 

    if output_type:
        batch_std[11] = np.std(y[:,0]) 
        batch_std[12] = np.std(y[:,1]) 
    else:
        batch_std[11] = np.std(y[:,:]) 

    global_mean = batch_mean
    global_std = batch_std
 
    return global_mean, global_std

####################################################################################################################################

####################################################################################################################################
def IQR(dist):
    return np.percentile(dist, 75) - np.percentile(dist, 25)

def compute_mean_IQR(batch_X, y,LSTM_recurrent_steps,ms_recurrent_steps,output_type):
    
    if output_type:
        global_mean = np.full((13), None)
        global_iqr = np.full((13), None)
        batch_mean = np.full((13), None)
        batch_iqr = np.full((13), None)
    else:
        global_mean = np.full((12), None)
        global_iqr = np.full((12), None)
        batch_mean = np.full((12), None)
        batch_iqr = np.full((12), None)

    batch_mean[0] = np.mean(batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_mean[1] = np.mean(batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_mean[2] = np.mean(batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_mean[3] = np.mean(batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_mean[4] = np.mean(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_mean[5] = np.mean(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_mean[6] = np.mean(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_mean[7] = np.mean(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_mean[8] = np.mean(batch_X[:,-3,:,:]) 
    batch_mean[9] = np.mean(batch_X[:,-2,:,:]) 
    batch_mean[10] = np.mean(batch_X[:,-1,:,:]) 

    if output_type:
        batch_mean[11] = np.mean(y[:,0]) 
        batch_mean[12] = np.mean(y[:,1]) 

 
    else:
        batch_mean[11] = np.mean(y[:,:]) 


    batch_iqr[0] = IQR(batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_iqr[1] = IQR(batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_iqr[2] = IQR(batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_iqr[3] = IQR(batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps-1,:,:]) 
    batch_iqr[4] = IQR(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_iqr[5] = IQR(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_iqr[6] = IQR(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_iqr[7] = IQR(batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps-1),:,:]) 
    batch_iqr[8] = IQR(batch_X[:,-3,:,:]) 
    batch_iqr[9] = IQR(batch_X[:,-2,:,:]) 
    batch_iqr[10] = IQR(batch_X[:,-1,:,:]) 

    if output_type:
        batch_iqr[11] = IQR(y[:,0]) 
        batch_iqr[12] = IQR(y[:,1]) 
    else:
        batch_iqr[11] = IQR(y[:,:]) 

    global_mean = batch_mean
    global_iqr = batch_iqr
 
    return global_mean, global_iqr

####################################################################################################################################

####################################################################################################################################


def normalize_min_max(batch_X, y,global_min,global_max,LSTM_recurrent_steps,ms_recurrent_steps,output_type):

    batch_X[:,0:0*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,0:0*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_min[0])/(global_max[0]-global_min[0])
    batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps:1*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps:1*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_min[1])/(global_max[1]-global_min[1])
    batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps:2*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps:2*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_min[2])/(global_max[2]-global_min[2])
    batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps:3*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps:3*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_min[3])/(global_max[3]-global_min[3])
    batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps:(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps:(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_min[4])/(global_max[4]-global_min[4]) 
    batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_min[5])/(global_max[5]-global_min[5]) 
    batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_min[6])/(global_max[6]-global_min[6]) 
    batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_min[7])/(global_max[7]-global_min[7]) 
    batch_X[:,-3,:,:] = (batch_X[:,-3,:,:]-global_min[8])/(global_max[8]-global_min[8]) 
    batch_X[:,-2,:,:] = (batch_X[:,-2,:,:]-global_min[9])/(global_max[9]-global_min[9]) 
    batch_X[:,-1,:,:] = (batch_X[:,-1,:,:]-global_min[10])/(global_max[10]-global_min[10]) 
    if output_type:
        y[:,0] = (y[:,0]-global_min[11])/(global_max[11]-global_min[11])
        y[:,1] = (y[:,1]-global_min[12])/(global_max[12]-global_min[12])
    else:
        y = (y-global_min[11])/(global_max[11]-global_min[11])

    return batch_X, y

####################################################################################################################################

####################################################################################################################################


def standardize_mean_std(batch_X, y,global_mean,global_std,LSTM_recurrent_steps,ms_recurrent_steps,output_type):

    batch_X[:,0:0*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,0:0*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_mean[0])/(global_std[0])
    batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps:1*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps:1*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_mean[1])/(global_std[1])
    batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps:2*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps:2*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_mean[2])/(global_std[2])
    batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps:3*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps:3*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_mean[3])/(global_std[3])
    batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps:(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps:(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_mean[4])/(global_std[4]) 
    batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_mean[5])/(global_std[5]) 
    batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_mean[6])/(global_std[6]) 
    batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_mean[7])/(global_std[7]) 
    batch_X[:,-3,:,:] = (batch_X[:,-3,:,:]-global_mean[8])/(global_std[8]) 
    batch_X[:,-2,:,:] = (batch_X[:,-2,:,:]-global_mean[9])/(global_std[9]) 
    batch_X[:,-1,:,:] = (batch_X[:,-1,:,:]-global_mean[10])/(global_std[10]) 
    if output_type:
        y[:,0] = (y[:,0]-global_mean[11])/(global_std[11])
        y[:,1] = (y[:,1]-global_mean[12])/(global_std[12])
    else:
        y = (y-global_mean[11])/(global_std[11])

    return batch_X, y

####################################################################################################################################

####################################################################################################################################


def robustscaling_mean_iqr(batch_X, y,global_mean,global_iqr,LSTM_recurrent_steps,ms_recurrent_steps,output_type):

    batch_X[:,0:0*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,0:0*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_mean[0])/(global_iqr[0])
    batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps:1*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,0*LSTM_recurrent_steps+LSTM_recurrent_steps:1*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_mean[1])/(global_iqr[1])
    batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps:2*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,1*LSTM_recurrent_steps+LSTM_recurrent_steps:2*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_mean[2])/(global_iqr[2])
    batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps:3*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:] = (batch_X[:,2*LSTM_recurrent_steps+LSTM_recurrent_steps:3*LSTM_recurrent_steps+LSTM_recurrent_steps,:,:]-global_mean[3])/(global_iqr[3])
    batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps:(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,3*LSTM_recurrent_steps+LSTM_recurrent_steps:(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_mean[4])/(global_iqr[4]) 
    batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_mean[5])/(global_iqr[5]) 
    batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_mean[6])/(global_iqr[6]) 
    batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps),:,:] = (batch_X[:,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps),:,:]-global_mean[7])/(global_iqr[7]) 
    batch_X[:,-3,:,:] = (batch_X[:,-3,:,:]-global_mean[8])/(global_iqr[8]) 
    batch_X[:,-2,:,:] = (batch_X[:,-2,:,:]-global_mean[9])/(global_iqr[9]) 
    batch_X[:,-1,:,:] = (batch_X[:,-1,:,:]-global_mean[10])/(global_iqr[10]) 
    if output_type:
        y[:,0] = (y[:,0]-global_mean[11])/(global_iqr[11])
        y[:,1] = (y[:,1]-global_mean[12])/(global_iqr[12])
    else:
        y = (y-global_mean[11])/(global_iqr[11])

    return batch_X, y

####################################################################################################################################

####################################################################################################################################

def monitor(histories, names, colors, early_stopping=1):
    assert len(histories) == len(names)
    assert len(histories) == len(colors)
    plt.figure(figsize=(15,6))
    for idx in range((len(histories))):
        plt.plot(histories[idx]['mse'][1:], label=names[idx]+' Training', alpha=.4, color=colors[idx], linestyle='--')
        plt.plot(histories[idx]['val_mse'][1:], label=names[idx]+' Validation', alpha=.8, color=colors[idx])   
    # plt.ylim(0.0075, 0.02)
    plt.title('Mean Squared Error')
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid(alpha=.3)
    plt.show()

####################################################################################################################################

####################################################################################################################################


def reshape_for_lstm(data, n_steps):
    n_vars = 8
    n_static_vars = 3
    batch_size, channels, height, width = data.shape

    # Get vars
    temporal_vars = []
    for i in range(n_vars):
        temporal_vars.append(data[:, i * n_steps:(i + 1) * n_steps, :, :])

    temporal_vars = np.stack(temporal_vars, axis=2)  # (batch_size, steps, variables, height, width)
    
    # Reshape to (batch_size, timesteps, height * width * variables)
    temporal_vars = temporal_vars.transpose(0, 1, 3, 4, 2)  # (batch_size, steps, height, width, variables)
    temporal_vars = temporal_vars.reshape(batch_size, n_steps, -1)  # (batch_size, steps, features)

    # Get static variables
    static_vars = data[:, -n_static_vars:, :, :]  # (batch_size, static_vars, height, width)
    static_vars = static_vars.reshape(batch_size, -1)  # (batch_size, static_vars * height * width)

    # Add static variables to time steps
    static_vars = np.repeat(static_vars[:, np.newaxis, :], n_steps, axis=1)  # (batch_size, steps, static_vars * height * width)
    lstm_input = np.concatenate((temporal_vars, static_vars), axis=2)  # (batch_size, steps, features + static_vars * height * width)
    
    return lstm_input # Shape: (batch_size, n_steps, vars * height * width)

####################################################################################################################################

####################################################################################################################################

def reshape_for_convlstm(data, n_steps):
    n_vars = 8
    n_static_vars = 3
    batch_size, channels, height, width = data.shape

    temporal_vars = []
    for i in range(n_vars):
        temporal_vars.append(data[:, i * n_steps:(i + 1) * n_steps, :, :])

    temporal_vars = np.stack(temporal_vars, axis=2)  # (batch_size, steps, variables, height, width)
    temporal_vars = temporal_vars.transpose(0, 1, 3, 4, 2)  # (batch_size, steps, height, width, variables)

    static_vars = data[:, -n_static_vars:, :, :]  # (batch_size, static_vars, height, width)


    static_vars = np.repeat(static_vars[:, np.newaxis, :, :, :], n_steps, axis=1)  # (batch_size, steps, static_vars, height, width)
    static_vars = static_vars.transpose(0, 1, 3, 4, 2)  # (batch_size, steps, height, width, static_vars)
    
    convlstm_input = np.concatenate((temporal_vars, static_vars), axis=-1)  # (batch_size, steps, height, width, channels)
    
    return convlstm_input


####################################################################################################################################

####################################################################################################################################

def build_dense_model(input_shape, output_shape,learning_rate,elastic_lambda,loss_name):
    
    # # Build the neural network layer by layer
    # input_layer = tfkl.Input(shape=input_shape, name='input_layer')
    # flatten_layer = tfkl.Flatten()(input_layer)
    # hidden_layer3 = tfkl.Dense(units=256, activation='leaky_relu' ,kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda), name='Hidden0')(flatten_layer)   
    # dropout = tfkl.Dropout(0.1)(hidden_layer3)  
    # hidden_layer3 = tfkl.Dense(units=128, activation='leaky_relu' ,kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda), name='Hidden1')(dropout)   
    # dropout = tfkl.Dropout(0.05)(hidden_layer3)  
    # hidden_layer3 = tfkl.Dense(units=64, activation='leaky_relu' ,kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda), name='Hidden2')(dropout)  
    # hidden_layer3 = tfkl.Dense(units=32, activation='leaky_relu' ,kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda), name='Hidden3')(hidden_layer3)  
    # output_layer = tfkl.Dense(units=output_shape[0], activation='leaky_relu', name='output')(hidden_layer3)         

    # # Connect input and output through the Model class
    # model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # # Compile the model
    # optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)
    # model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=optimizer, metrics=['mse'])

    # # Return the model
    # return model

    
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')
    flatten_layer = tfkl.Flatten()(input_layer)
    hidden_layer3 = tfkl.Dense(units=256, activation='leaky_relu', name='Hidden0')(flatten_layer)   
    hidden_layer3 = tfkl.Dense(units=128, activation='leaky_relu', name='Hidden1')(hidden_layer3)   
    hidden_layer3 = tfkl.Dense(units=64, activation='leaky_relu', name='Hidden2')(hidden_layer3)  
    hidden_layer3 = tfkl.Dense(units=32, activation='leaky_relu', name='Hidden3')(hidden_layer3)  
    output_layer = tfkl.Dense(units=output_shape[0], activation='leaky_relu', name='output')(hidden_layer3)         

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)
    if loss_name == "mse":
        model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=optimizer, metrics=['mse'])
    if loss_name == "mcql":
        model.compile(loss=cl.monotone_composite_quantile_loss, optimizer=optimizer, metrics=['mse'])
 

    # Return the model
    return model

####################################################################################################################################

####################################################################################################################################

def build_CONV_model(input_shape, output_shape,learning_rate,elastic_lambda,seed,loss_name):
    
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    conv1 = tfkl.Conv2D(
        filters=128,
        kernel_size=(int(np.round(input_shape[0]/2)), int(np.round(input_shape[1]/2))),
        kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda),
        strides = (1, 1),
        padding = 'same',
        activation = 'leaky_relu',
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(input_layer)

    conv2 = tfkl.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda),
        strides = (1, 1),
        padding = 'same',
        activation = 'leaky_relu',
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(conv1)
    
    pool1 = tfkl.MaxPooling2D(pool_size = (2, 2))(conv2)

    conv3 = tfkl.Conv2D(
        filters=128,
        kernel_size=(2, 2),
        kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda),
        strides = (1, 1),
        padding = 'same',
        activation = 'leaky_relu',
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(pool1)

    conv4 = tfkl.Conv2D(
        filters=64,
        kernel_size=(2, 2),
        kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda),
        strides = (1, 1),
        padding = 'same',
        activation = 'leaky_relu',
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(conv3)

    pool2 = tfkl.MaxPooling2D(pool_size = (2, 2))(conv4)


    flattening_layer = tfkl.Flatten(name='Flatten')(pool2)
    
    dropout = tfkl.Dropout(0.1)(flattening_layer)  

    classifier_layer = tfkl.Dense(
        units=64, 
        name='Classifier', 
        activation='leaky_relu',
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(dropout)

    output_layer = tfkl.Dense(units=output_shape[0], activation='leaky_relu', name='output')(classifier_layer)         

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)

    if loss_name == "mse":
        model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=optimizer, metrics=['mse'])
    if loss_name == "mcql":
        model.compile(loss=cl.monotone_composite_quantile_loss, optimizer=optimizer, metrics=['mse'])

    # Return the model
    return model

####################################################################################################################################

####################################################################################################################################


def build_LSTM_model(input_shape, output_shape,learning_rate,elastic_lambda,seed,loss_name):
    
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')
    x = tfkl.LSTM(128, activation='leaky_relu',dropout=0.05, 
                  return_sequences=True, 
                  name='lstm')(input_layer)
    # x = tfkl.BatchNormalization(axis=-1)(x)
    x = tfkl.LSTM(64, activation='leaky_relu',dropout=0.05, 
                  return_sequences=True, 
                  name='lstm_2')(x)
    # x = tfkl.BatchNormalization(axis=-1)(x)
    x = tfkl.LSTM(32, activation='leaky_relu',dropout=0.05,  
                  return_sequences=True, 
                  name='lstm_3')(x)
    # x = tfkl.BatchNormalization(axis=-1)(x)
    flattening_layer = tfkl.Flatten(name='Flatten')(x)
    
    dropout = tfkl.Dropout(0.08)(flattening_layer)  

    classifier_layer = tfkl.Dense(
        units=32, 
        name='Classifier', 
        activation='leaky_relu',
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(dropout)

    output_layer = tfkl.Dense(units=output_shape[0], activation='linear', name='output')(classifier_layer)      
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')
    # Compile the model
    optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)

    if loss_name == "mse":
        model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=optimizer, metrics=['mse'])
    if loss_name == "mcql":
        model.compile(loss=cl.monotone_composite_quantile_loss, optimizer=optimizer, metrics=['mse'])
 

    # Return the model
    return model

####################################################################################################################################

####################################################################################################################################

def build_Conv3D_model(input_shape, output_shape,learning_rate,elastic_lambda,seed,loss_name):
    
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    x = tfkl.Conv3D(
        filters=128,
        kernel_size=(int(np.round(input_shape[0]/2)), int(np.round(input_shape[1]/2)), int(np.round(input_shape[2]/2))),
        strides=(1, 1, 1),
        padding = 'valid',
        activation = 'leaky_relu',
        # kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda),
        # bias_regularizer=tfk.regularizers.L1L2(elastic_lambda)
    )(input_layer)

    if input_shape[2] >= 7:
        x = tfkl.Conv3D(
            filters=128,
            kernel_size=(1, 2, 3),
            strides=(1, 1, 1),
            padding = 'valid',
            activation = 'leaky_relu',
            # kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda),
            # bias_regularizer=tfk.regularizers.L1L2(elastic_lambda)
        )(x)

    x = tfkl.Conv3D(
        filters=128,
        kernel_size=(2, 2, 2),
        strides=(1, 1, 1),
        padding = 'valid',
        activation = 'leaky_relu',
        # kernel_regularizer=tfk.regularizers.L1L2(elastic_lambda),
        # bias_regularizer=tfk.regularizers.L1L2(elastic_lambda)
    )(x)

    flattening_layer = tfkl.Flatten(name='Flatten')(x)

    output_layer = tfkl.Dense(units=output_shape[0], activation='leaky_relu', name='output')(flattening_layer)         

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=optimizer, metrics=['mse'])

    # Return the model
    return model

