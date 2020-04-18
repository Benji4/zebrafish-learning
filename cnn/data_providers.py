import torch
import numpy as np
import time
import os
import h5py
from torch.utils import data


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, all_file, num_samples_per_worker):
        'Initialization'
        # self.labels = labels
        self.list_IDs = list_IDs
        self.all_file = all_file
        self.num_samples_per_worker = num_samples_per_worker

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Get the worker_id to find the right file, and the index within the worker file:
        worker_id = ID // self.num_samples_per_worker
        idx = ID % self.num_samples_per_worker

        # Load data and get label
        with h5py.File(self.all_file, 'r') as f:
            # load original frames = spatial input:
            X = f['worker-{}-inputs_orig'.format(worker_id)][idx]
            X = X[np.newaxis]  # empty axis for channels

            # load flow for temporal input:
            X_flow = f['worker-{}-inputs_flow'.format(worker_id)][idx]
            # ss = time.time()
            minmax = f['worker-{}-minmax'.format(worker_id)][idx]
            # print("Loading minmax took {0:.2f} secs".format(time.time() - ss)) # about 0.01 secs

            # load labels:
            y = np.uint8(f['worker-{}-targets'.format(worker_id)][idx])



        return X, X_flow, y, minmax