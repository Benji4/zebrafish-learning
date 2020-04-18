import numpy as np
import os
import torch
import h5py
from collections import OrderedDict
import time
import innvestigate.applications.imagenet
import innvestigate
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D)

seed = 462019
tf.set_random_seed(seed)
np.random.seed(seed)

def reverse_rescale(X, min, max):
    """ in-place for minimal RAM consumption """
    ### Meaning:  X = min + X * (max-min) / 255
    max -= min
    max /= 255
    X *= max[...,np.newaxis,np.newaxis]  # add two empty axes for broadcasting
    X += min[...,np.newaxis,np.newaxis]
    return X

# Pre-trained initializers
def cnn_m_2048(inits):
    model = Sequential([
        Conv2D(96, kernel_size=7, strides=2, padding='valid', activation="relu", use_bias=True, kernel_initializer=inits[0], bias_initializer=inits[1]), # data_format = "channels_first"
        MaxPooling2D(pool_size=3, strides=2),
        ZeroPadding2D(padding=1),
        Conv2D(256, kernel_size=5, strides=2, padding='valid', activation="relu", use_bias=True, kernel_initializer=inits[2], bias_initializer=inits[3]),
        MaxPooling2D(pool_size=3, strides=2),
        ZeroPadding2D(padding=1),
        Conv2D(512, kernel_size=3, strides=1, padding='valid', activation="relu", use_bias=True, kernel_initializer=inits[4], bias_initializer=inits[5]),
        ZeroPadding2D(padding=1),
        Conv2D(512, kernel_size=3, strides=1, padding='valid', activation="relu", use_bias=True, kernel_initializer=inits[6], bias_initializer=inits[7]),
        ZeroPadding2D(padding=1),
        Conv2D(512, kernel_size=3, strides=1, padding='valid', activation="relu", use_bias=True, kernel_initializer=inits[8], bias_initializer=inits[9]),
        MaxPooling2D(pool_size=3, strides=2),

        Flatten(),
        Dense(4096, activation="relu", use_bias=True, kernel_initializer=inits[10], bias_initializer=inits[11]),
        Dense(2048, activation="relu", use_bias=True, kernel_initializer=inits[12], bias_initializer=inits[13]),
        Dense(2, activation="relu", use_bias=True, kernel_initializer=inits[14], bias_initializer=inits[15])
        # Dense(2, activation="relu", use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform')
    ])

    return model

# Random initializers
# def cnn_m_2048_rnd():
#     model = Sequential([
#         Conv2D(96, kernel_size=7, strides=2, padding='valid', activation="relu", use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'), # data_format = "channels_first"
#         MaxPooling2D(pool_size=3, strides=2),
#         ZeroPadding2D(padding=1),
#         Conv2D(256, kernel_size=5, strides=2, padding='valid', activation="relu", use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'),
#         MaxPooling2D(pool_size=3, strides=2),
#         ZeroPadding2D(padding=1),
#         Conv2D(512, kernel_size=3, strides=1, padding='valid', activation="relu", use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'),
#         ZeroPadding2D(padding=1),
#         Conv2D(512, kernel_size=3, strides=1, padding='valid', activation="relu", use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'),
#         ZeroPadding2D(padding=1),
#         Conv2D(512, kernel_size=3, strides=1, padding='valid', activation="relu", use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'),
#         MaxPooling2D(pool_size=3, strides=2),
#
#         Flatten(),
#         Dense(4096, activation="relu", use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'),
#         Dense(2048, activation="relu", use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'),
#         Dense(2, activation="relu", use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform')
#     ])
#
#   return model


if __name__ == '__main__':
    ss = time.time()

    # Run multiple analyses with different techniques, weights, and datasets, specified here. Names correspond to the heatmaps to be obtained from iNNvestigate, corresponding weights obtained before from CNN training corresponding datasets:
    abbreviations = ['14f_gui', '12f_gui'] # ['14f_sens', '12f_sens', '14f_dec', '12f_dec']
    weights_abbreviations = ['14f','12f'] # ['14f','12f','14f','12f']
    datas = ['13f', '9f'] # ['13f', '9f', '13f', '9f']

    for run in range(len(abbreviations)):
        abbr = abbreviations[run]
        weights_abbr = weights_abbreviations[run]
        data = datas[run]

        pth_path = 'data/z_{}.pth'.format(weights_abbr) # weights to be analyzed  # TODO path to file
        input_file = '/disk/scratch/{}/all.hdf5'.format(data) # data to be analyzed  # TODO path to file
        outfile = '/disk/scratch/analysis_{}/analysis.hdf5'.format(abbr) # TODO path to output file

        worker_ids = [32, 33, 34, 35, 36, 37] # test set


        print("Loading weights in pytorch ...")
        state = torch.load(pth_path, map_location='cpu')
        spatialnet = OrderedDict()
        temporalnet = OrderedDict()
        for i, s in enumerate(state['network']):
            s_new = s.replace("model.module.", "")
            if s_new.find("spatial_stream.") == 0:
                s_new = s_new.replace("spatial_stream.", "")
                spatialnet[s_new] = state['network'][s]
            elif s_new.find("temporal_stream.") == 0:
                s_new = s_new.replace("temporal_stream.", "")
                temporalnet[s_new] = state['network'][s]
        spatialnet_list = list(spatialnet.items())
        temporalnet_list = list(temporalnet.items())


        # Initialize network:
        inits_spatial = []
        inits_temporal = []
        for i in range(16): # hard-coded weight initializers
            # print(spatialnet_list[i][0])
            inits_spatial.append(tf.constant_initializer(np.asarray(spatialnet_list[i][1].numpy().T, dtype=np.float32)))
            inits_temporal.append(tf.constant_initializer(np.asarray(temporalnet_list[i][1].numpy().T, dtype=np.float32)))

        # Allocate space. Will not be fully filled if not all inputs can be kept.
        # We later count how many samples were processed correctly.
        n_tot = 3420 # 190 * 18
        spatial_out_shape = (n_tot, 224, 224)
        temporal_out_shape = (n_tot, 85, 224, 224)
        with h5py.File(outfile, 'w') as f:
            f.create_dataset("spatial", spatial_out_shape, dtype=np.float32)
            f.create_dataset("temporal", temporal_out_shape, dtype=np.float32)
            f.create_dataset("kept_idxs", (n_tot,), dtype=np.uint32)

        start = 0
        kept_idxs_tot = 0
        offset = 0
        for worker_id in (worker_ids):
            print("Loading hdf5-file ...")
            with h5py.File(input_file, 'r') as f:
                x = f['worker-{}-inputs_orig'.format(worker_id)][:]
                x_flow = f['worker-{}-inputs_flow'.format(worker_id)][:]
                minmax = f['worker-{}-minmax'.format(worker_id)][:]

            x = x[:,np.newaxis] # empty axis for channels

            n = len(x) # number of samples analyzed here

            x_flow = reverse_rescale(x_flow.astype(np.float32), minmax[..., 0], minmax[..., 1])
            print(x.shape, x_flow.shape, minmax.shape)

            # For some samples, the analyzer returns empty results. We omit these samples.
            # This would make it work, but we rather omit the samples which the analyzer has problems analyzing.
            # x_flow += np.random.normal(size=x_flow.shape, scale=1) # 0.000001 does not work, but 1 does.
            kept_idxs = []

            ################# SPATIAL ####################

            print("Generating spatial analysis ...")

            input_range = (0,255)


            model_s = cnn_m_2048(inits_spatial)
            model_s.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model_s.build(input_shape=(n, 1, 224, 224))
            if abbr == '14f_sens' or abbr == '12f_sens':
                opt_params = {"postprocess": "abs"}
                analyzer = innvestigate.create_analyzer("gradient", model_s, **opt_params)
            elif abbr == '14f_dec' or abbr == '12f_dec':
                analyzer = innvestigate.create_analyzer("deconvnet", model_s)
            elif abbr == '14f_gui' or abbr == '12f_gui':
                analyzer = innvestigate.create_analyzer("guided_backprop", model_s)
            elif abbr == 'deep_taylor':
                opt_params = {"low": input_range[0], "high": input_range[1]}
                analyzer = innvestigate.create_analyzer("deep_taylor.bounded", model_s, **opt_params)
            else:
                print('wrong abbreviation')
                exit(1)
            analysis_s = analyzer.analyze(x)
            analysis_s = np.squeeze(analysis_s)


            ################# TEMPORAL ####################

            print("Generating temporal analysis ...")

            input_range = (np.min(x_flow), np.max(x_flow))
            print("input_range {}".format(input_range))


            model_t = cnn_m_2048(inits_temporal)
            model_t.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model_t.build(input_shape=(n, 170, 224, 224))  # we have to use 170 here!
            if abbr == '14f_sens' or abbr == '12f_sens':
                opt_params = {"postprocess": "abs"}
                analyzer = innvestigate.create_analyzer("gradient", model_t, **opt_params)
            elif abbr == '14f_dec' or abbr == '12f_dec':
                analyzer = innvestigate.create_analyzer("deconvnet", model_t)
            elif abbr == '14f_gui' or abbr == '12f_gui':
                analyzer = innvestigate.create_analyzer("guided_backprop", model_t)
            elif abbr == 'deep_taylor':
                opt_params = {"low": input_range[0], "high": input_range[1]}
                analyzer = innvestigate.create_analyzer("deep_taylor.bounded", model_t, **opt_params)
            else:
                print('wrong abbreviation')
                exit(1)
            analysis_t = analyzer.analyze(x_flow)

            # print("analysis_t {}".format(analysis_t))
            analysis_t = analysis_t.reshape(len(analysis_t), 85, 2, 224, 224)
            analysis_t = np.mean(analysis_t, axis=2) # mean of relevance over x and y components

            ################# OUTPUT #################

            # keep only samples which did not return an empty analysis in either of the streames
            kept_idxs = []  # kept indices
            for i in range(n):
                if not np.all(np.isclose(analysis_s[i], 0)) and not np.all(np.isclose(analysis_t[i], 0)):
                    kept_idxs.append(i)

            kept_idxs = np.array(kept_idxs)
            out_n = len(kept_idxs)
            end = start + out_n

            # Write to hdf5 file:
            print("Outputting spatial and temporal analysis to {} ...".format(outfile))
            print("Writing to {}:{}".format(start,end))
            with h5py.File(outfile, 'a') as f:
                f['spatial'][start:end] = analysis_s[kept_idxs]
                f['temporal'][start:end] = analysis_t[kept_idxs]
                f['kept_idxs'][start:end] = (kept_idxs + offset)

            start += out_n
            kept_idxs_tot += out_n
            offset += n

        print("############################################################")
        print("We removed {} samples, which leaves in total {} samples.".format(n_tot - kept_idxs_tot, kept_idxs_tot))
        print("Done in {0:.2f} secs".format(time.time() - ss))