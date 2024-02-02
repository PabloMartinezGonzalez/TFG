"""
Created on Mon Dec 4 09:00:00

@author: adrian.perez

Script to generate np data for breath and warp aligned.
"""
#%% Import and load cell

import numpy as np
import plotly.io as pio
import torch
from sklearn.preprocessing import scale

pio.renderers.default = "browser"
import wfdb
from scipy import signal, stats
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dtype = torch.float64
torch.set_default_dtype(dtype)

def butt_denoise(d_signal, freq, freq_filter):
    fs = freq  # Sampling frequency
    fc = freq_filter  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    output = signal.filtfilt(b, a, d_signal)
    return output
print("carga de datos")
rec_ = 'f1y01'
rec = rec_

dir = rec+"/"
file_path = dir + rec
record = wfdb.rdrecord(file_path, return_res=32, physical=False)
N_0 = 1
#N = 6000
N = -1
freq_filter_pred = 5.0
labels = wfdb.rdann(file_path, 'ecg', return_label_elements=['symbol']).symbol[N_0:N]
annotation = wfdb.rdann(file_path, 'ecg').sample[N_0:N]
record = wfdb.rdrecord(file_path, return_res=32, physical=False)

d_signal = record.d_signal
#samples = [0, 190]
data_breath = d_signal[:,0]
data_ecg = d_signal[:,1]
sub_annotation = []

#Filter signals.
freq = 250
freq_filter = 1
#APNEA
freq_filter = 50
data_breath = butt_denoise(data_breath, freq, freq_filter)
data_ecg = butt_denoise(data_ecg, freq, 30)


#Now read warps
x_warps = np.load(dir + 'x_warps_'+rec_+'.npy')
x_warps_ = []
for x in x_warps[N_0:N]:
    x_warps_.append(x.T[0])
#First heartbeat is used as model, so we don't have warp for it, x_warps starts in annotation[1]

#Now we are going to plot the situation that we have
L = len(x_warps[0])
samples_ecg = [0,L]

#Choose the part of the warp you want to work with.
#This numbers must surround 87 that is the peak of QRS and is where most information is available.
#size_chosen = [10,200]
#size_chosen = [86,87]
#size_chosen = [75,95]
#size_chosen = [10,170]
size_chosen = [20,160]
#size_chosen = [50,115] #QRS part
real_size = size_chosen

#This is the part of the breathing signal we want to compute with the net
#In this case  with LSTSQ they must have the same dimension
size_chosen_out = size_chosen
#size_chosen_out = [65, 135]
#size_chosen_out = [0, 210]
#size_chosen_out = [75,95]
#size_chosen_out = [86,87]
#size_chosen_out = [20, 160]

x_warps = []
for xw in x_warps_:
    x_warps.append(xw[size_chosen[0]:size_chosen[1]])

databreath = []
databr = scale(data_breath)
for i in annotation:
    databreath.append(np.array(databr[i-87+size_chosen_out[0]:i-87+size_chosen_out[1]], dtype=np.float64))

#Then define the training data
#t_train = 200
t_test = len(x_warps)-1
#t_test = t_train + 30
t_train = t_test
train_y = []
train_x = x_warps
for j, i in enumerate(annotation[1:]):
    i1 = i-87+size_chosen_out[0]
    i2 = i-87+size_chosen_out[1]
    train_y.append(databr[i1:i2])
train_y = torch.from_numpy(np.array(train_y))
train_x = torch.from_numpy(np.array(train_x))

train_x_domain = []
train_y_domain = []
for j, i in enumerate(annotation[1:]):
    i1 = i-87+real_size[0]
    i2 = i-87+real_size[1]
    train_x_domain.append(list(range(i1, i2)))
    train_y_domain.append(list(range(i - 87 + size_chosen_out[0], i - 87 + size_chosen_out[1])))


## Part for latent ode export data
warp = train_x.numpy().T
time = np.array(train_x_domain).T
print(time.shape)
print(warp.shape)

min_length = min(time.shape[1], warp.shape[1])
time = time[:, :min_length]
warp = warp[:, :min_length]
data_full = np.stack([time, warp]).T

single = databr[train_x_domain[0][0]:train_x_domain[20][-1]]
single_time = np.arange(train_x_domain[0][0],train_x_domain[20][-1])
save = True

dir_data = dir + "data/"
if not os.path.exists(dir_data):
    	os.makedirs(dir_data)

np.save(dir_data + 'time_warp.npy', data_full)
np.save(dir_data + 'breath.npy', databreath[:len(data_full)])
np.save(dir_data + 'breath_single.npy', np.stack([single, single_time]).T)

