###########################
# Latent ODEs for Irregularly-Sampled Time Series
# 	Application into warp breathing prediction
# Author: Adrian Perez Herrero
###########################


from __future__ import absolute_import, division
from __future__ import print_function
import os
import matplotlib
#matplotlib.use('Agg')

import numpy as np
import numpy.random as npr
from scipy.special import expit as sigmoid
import pickle
import matplotlib.pyplot as plt
import matplotlib.image
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import scale
import lib.utils as utils
dtype = torch.float32

# ======================================================================================

def get_data_min_max(records, device=torch.device("cpu")):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	data_min, data_max = None, None
	inf = torch.Tensor([float("Inf")])[0].to(device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf.to(device))
				batch_max.append(-inf.to(device))
			else:
				batch_min.append(torch.min(non_missing_vals).to(device))
				batch_max.append(torch.max(non_missing_vals).to(device))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min.to(device), data_max.to(device)

#This class is to generate data associated with warp and breath. It has different methods to structure data to carry
#different experiments. It is necessary to have data in concrete folder.
class WarpTimeSeries(object):

	params = [ 'Warp' ]
	params_dict = {k: i for i, k in enumerate(params)}

	labels = [ "Breath" ]
	labels_dict = {k: i for i, k in enumerate(labels)}

	def __init__(self, root, n_samples=300, device = torch.device("cpu")):
		self.device = device
		self.root = root

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You have to put warp.npy and breath.npy in your data directory.')

		data_file = self.data_file

		if device == torch.device("cpu"):
			data = np.load(os.path.join(self.processed_folder, data_file))
			self.labels = torch.from_numpy(np.load(os.path.join(self.processed_folder, self.label_file))).to(device)
			#first_example = np.load(os.path.join(self.processed_folder, self.single_data_file))
			first_example = None
			self.data = self.process_data(data, self.labels, device, n_samples, first_example, dat_included='all')
		else:
			data = np.load(os.path.join(self.processed_folder, data_file))
			self.labels = torch.from_numpy(np.load(os.path.join(self.processed_folder, self.label_file)))
			#first_example = np.load(os.path.join(self.processed_folder, self.single_data_file))
			first_example = None
			self.data = self.process_data(data, self.labels, device, n_samples, first_example, dat_included='all')

		#if n_samples is not None:
		#	self.data = self.data[:n_samples]
		#	self.labels = self.labels[:n_samples]

	def _check_exists(self):
		files = ["breath", "time_warp"]
		for filename in files:

			if not os.path.exists(
					os.path.join(self.processed_folder,
								 filename.split('.')[0] + '.npy')
			):
				return False
		return True

	#Method to process data where one observation is a warp.
	#Here breath is conceived as labels.
	# def process_data(self, data, labels, device):
	# 	data_new = []
	# 	for i, d in enumerate(data):
	# 		dat = []
	# 		dat.append(torch.tensor(i, dtype=dtype).to(device)) #Identificator of example
	# 		dat.append(torch.from_numpy(d[:,0]/250.0).type(dtype).to(device)) # (t) Sample frequency
	# 		#dat.append(torch.from_numpy(np.atleast_2d(d[:,1]).T).type(dtype).to(device)) #(obs)
	# 		dat.append(torch.from_numpy(np.stack([d[:,1], labels[i]]).T).type(dtype).to(device)) #(obs)
	# 		dat.append(torch.atleast_2d(torch.ones(d.shape[0], dtype=dtype)).T.to(device)) #(mask)
	# 		#dat.append(torch.tensor([labels[i]]).T.type(dtype).to(device)) #(labels)
	# 		data_new.append(tuple(dat))
	# 	return data_new

	#Method to process data where one observation is a warp and labels are not important.
	#Warp and breath are all in same vector.
	# def process_data(self, data, labels, device):
	# 	data_new = []
	# 	for i, d in enumerate(data):
	# 		dat = []
	# 		t = d[:,0]/250.0
	# 		#t = (t - np.min(t))/(np.max(t) - np.min(t))
	# 		dat.append(torch.tensor(i, dtype=dtype).to(device)) #Identificator of example
	# 		dat.append(torch.from_numpy(t).type(dtype).to(device)) # (t) Sample frequency
	# 		#dat.append(torch.from_numpy(np.atleast_2d(d[:,1]).T).type(dtype).to(device)) #(obs)
	# 		dat.append(torch.from_numpy(np.stack([d[:,1], labels[i]]).T).type(dtype).to(device)) #(obs)
	# 		dat.append(torch.atleast_2d(torch.ones(dat[-1].shape, dtype=dtype)).to(device)) #(mask)
	# 		dat.append(torch.tensor(np.repeat(1,d.shape[0])).type(dtype).to(device)) #(labels)
	# 		data_new.append(tuple(dat))
	# 	return data_new

	# Method to fil a bunch of data into one observation.
	def process_data(self, data, labels, device, n_samples, first_example=None, dat_included='all'):
		data_new = []
		rep = False
		if first_example is not None:
			max_std = np.max(first_example[:,1] - first_example[0,1])
			min_std = np.min(first_example[:, 1] - first_example[0,1])
		else:
			time = data[:n_samples, :, 0]
			time = time - time[0]
			max_std = np.max(time)
			min_std = np.min(time)
		#First one complete.
		if first_example is not None:
			i=0
			time_w = data[i * n_samples:(i + 1) * n_samples, :, 0] - data[0,0,0]
			time_w = torch.from_numpy(time_w.reshape((-1))).type(dtype).to(device)
			time = first_example[:,1] - first_example[0,1]
			time = torch.from_numpy(time.reshape((-1))).type(dtype).to(device)
			mask_w = [1 if j in time_w else 0 for j in time]
			obs = data[i * n_samples:(i + 1) * n_samples, :, 1].reshape(-1)
			warp_w = np.zeros(time.shape)
			warp_w[np.where(np.array(mask_w) == 1)] = obs
			time = (time - min_std) / (max_std - min_std)
			obs = np.stack([first_example[:,0], np.array(warp_w)]).T
			obs = torch.from_numpy(obs.reshape((-1, 2))).type(dtype).to(device)
			# obs = labels[i*n_samples:(i+1)*n_samples].reshape((-1,1)).type(dtype).to(device)
			id = torch.tensor(0).type(dtype).to(device)
			mask = torch.from_numpy(np.stack([np.ones(time.shape), mask_w]).T).type(dtype).to(device)
			labs = torch.ones(obs.shape[0]).type(dtype).to(device)
			data_new.append(tuple([id, time, obs, mask, labs]))
		#for i in range(10, 110):
		for i in range(1,100):
			#time = data[:n_samples,:,0]
			#time = (time - torch.min(time)) / (torch.max(time) - torch.min(time))
			time = data[i * n_samples:(i + 1) * n_samples, :, 0]
			obs = data[i*n_samples:(i+1)*n_samples,:,1]
			obs = np.stack([labels[i*n_samples:(i+1)*n_samples].numpy().T, obs.T]).T
			obs = torch.from_numpy(obs.reshape((-1,2))).type(dtype).to(device)
			#obs = labels[i * n_samples:(i + 1) * n_samples].numpy()
			#obs = torch.from_numpy(obs.reshape((-1, 1))).type(dtype).to(device)
			mask = torch.ones(obs.shape).type(dtype).to(device)
			if dat_included == 'breath':
				#obs[:, 1] = torch.zeros(obs.shape[0]).type(dtype).to(device)
				mask[:, 1] = torch.zeros(obs.shape[0]).type(dtype).to(device)
				for i in range(10):
					mask[i, 1] = + 1.0
			elif dat_included == 'warp':
				#obs[:, 0] = torch.zeros(obs.shape[0]).type(dtype).to(device)
				mask[:, 0] = torch.zeros(obs.shape[0]).type(dtype).to(device)
				for i in range(10):
					mask[i, 0] = + 1.0
			# if not rep:
			# 	time = data[i * n_samples:(i + 1) * n_samples, :, 0]  # /250.0
			# 	obs = torch.from_numpy(scale(labels[i*n_samples:(i+1)*n_samples])).reshape((-1,1)).type(dtype).to(device)
			# else:
			# 	time = data[:n_samples, :, 0]  # /250.0
			# 	obs = torch.from_numpy(scale(labels[:n_samples])).reshape((-1, 1)).type(dtype).to(device)
			time = torch.from_numpy(time.reshape((-1))).type(dtype).to(device)
			time = (time - time[0])/250.0
			#time = (time - min_std) / (max_std - min_std)
			#id = torch.tensor(i).type(dtype).to(device)
			id = str(i)
			labs = torch.ones(obs.shape[0]).type(dtype).to(device)
			data_new.append(tuple([id,time,obs,mask,labs]))
		return data_new

	# Method to process data where one observation is a single point of warping.
	# def process_data(self, data, labels, device):
	# 	# Flatten the data and labels for easier processing
	# 	flat_data = [item for sublist in data for item in sublist]
	# 	flat_labels = [item for sublist in labels for item in sublist]
	#
	# 	# Convert data to numpy arrays for vectorized operations
	# 	data_0 = np.array([[d[0]] for d in flat_data]) / 250.0
	# 	data_1 = np.array([[d[1]] for d in flat_data])
	#
	# 	# Convert labels to a tensor if they are not already
	# 	if not isinstance(flat_labels[0], torch.Tensor):
	# 		flat_labels = [torch.tensor([label]) for label in flat_labels]
	#
	# 	# Convert to tensors and move to device
	# 	data_0_tensor = torch.tensor(data_0, dtype=torch.float32).unsqueeze(1).to(device)
	# 	data_1_tensor = torch.tensor(data_1, dtype=torch.float32).unsqueeze(1).to(device)
	# 	labels_tensor = torch.stack(flat_labels).unsqueeze(1).to(device)
	#
	# 	# Create IDs and Ones tensor
	# 	ids = torch.arange(len(flat_data), dtype=torch.float32).to(device).unsqueeze(1)
	# 	ones_tensor = torch.ones_like(data_1_tensor)
	#
	# 	# Combine into final data structure
	# 	data_new = [(id_, d0, d1, one, label) for id_, d0, d1, one, label in
	# 				zip(ids, data_0_tensor, data_1_tensor, ones_tensor, labels_tensor)]
	#
	# 	return data_new

	@property
	def processed_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'processed')

	@property
	def label_file(self):
		return 'breath.npy'

	@property
	def data_file(self):
		return 'time_warp.npy'

	@property
	def single_data_file(self):
		return 'breath_single.npy'

	def __getitem__(self, index):
		return self.data[index]

	def get_dataset(self):
		return self.data

	def get_labels(self):
		return self.labels
	def __len__(self):
		return len(self.data)

	def get_label(self, record_id):
		return self.labels[record_id]

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)
		return fmt_str

	def visualize(self, timesteps, data, mask, plot_name):
		width = 15
		height = 15

		non_zero_attributes = (torch.sum(mask, 0) > 2).numpy()
		non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
		n_non_zero = sum(non_zero_attributes)

		mask = mask[:, non_zero_idx]
		data = data[:, non_zero_idx]

		params_non_zero = [self.params[i] for i in non_zero_idx]
		params_dict = {k: i for i, k in enumerate(params_non_zero)}

		n_col = 3
		n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
		fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

		# for i in range(len(self.params)):
		for i in range(n_non_zero):
			param = params_non_zero[i]
			param_id = params_dict[param]

			tp_mask = mask[:, param_id].long()

			tp_cur_param = timesteps[tp_mask == 1.]
			data_cur_param = data[tp_mask == 1., param_id]

			ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(), marker='o')
			ax_list[i // n_col, i % n_col].set_title(param)

		fig.tight_layout()
		fig.savefig(plot_name)
		plt.close(fig)

#Esta funcion identifica el vector de tiempos completo para el batch determinado. Identifica donde hay observaciones y
# genera un diccionario con estos datos ordenados. Además tiene unas funciones de normalizacion de valores.
def variable_time_collate_fn_warp(batch, args, device=torch.device("cpu"), data_type="train",
							 data_min=None, data_max=None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][2].shape[1]
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	combined_tt = combined_tt.to(device)

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D], dtype=dtype).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D], dtype=dtype).to(device)

	combined_labels = None
	N_labels = 1

	combined_labels = torch.zeros(len(batch), N_labels, dtype=dtype) + torch.tensor(float('nan'))
	combined_labels = combined_labels.to(device=device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		labels = None
		tt = tt.to(device)
		vals = vals.to(device)
		mask = mask.to(device)
		if labels is not None:
			labels = labels.to(device)

		indices = inverse_indices[offset:offset + len(tt)]
		offset += len(tt)

		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask

		if labels is not None:
			combined_labels[b] = labels

	combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask,
													  att_min=data_min, att_max=data_max)

	#if torch.max(combined_tt) != 0.:
	#	combined_tt = combined_tt / torch.max(combined_tt)

	data_dict = {
		"data": combined_vals,
		"time_steps": combined_tt,
		"mask": combined_mask,
		"labels": combined_labels}

	data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
	return data_dict

if __name__ == '__main__':
	torch.manual_seed(1991)

	dataset = WarpTimeSeries('data/warp')
	dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn)
	print(dataloader.__iter__().next())



