import numpy as np
import torch
import os
import pickle
import scipy.io
from collections import OrderedDict
np.random.seed(462019)

def extract_weights_from_matlab(matlab_file):
    mat = scipy.io.loadmat(matlab_file)
    layers = mat["layers"]
    indices = np.array([0, 4, 8, 10, 12, 15, 17, 19])  # hard-coded indices of the layers we want to extract
    weights = {}
    for i in indices:
        layer_name = layers[0][i][0][0][0][0]
        #     print(layer_name)
        layer_weights = np.squeeze(layers[0][i][0][0][2][0][0].T)  # transpose to fit to pytorch model
        layer_bias    = np.squeeze(layers[0][i][0][0][2][0][1])
        if i == 15:  # the layer fc6 is special, because we need to subsample from its weights
            layer_weights = layer_weights[:, :, :5, :5].reshape(4096, 12800)
        if i == 19:  # the final layer must be broken down to 2 classes, we do this by averaging
            tmp = layer_weights
            layer_weights = np.empty((2, tmp.shape[1]))
            layer_weights[0] = np.mean(tmp[:500], axis=0)
            layer_weights[1] = np.mean(tmp[500:], axis=0)

            tmp = layer_bias
            layer_bias = np.empty(2)
            layer_bias[0] = np.mean(tmp[:500], axis=0)
            layer_bias[1] = np.mean(tmp[500:], axis=0)

        print("{}: {}, {}".format(layer_name, layer_weights.shape, layer_bias.shape))
        weights[layer_name] = layer_weights
        weights["{}.bias".format(layer_name)] = layer_bias

    # pickle.dump(weights, open(os.path.join(model_save_dir, weights_outfile), "wb"))
    return weights

# constructs an init state using pretrained weights
def get_init_state(weights):
    state = dict()
    state['current_epoch_idx'] = 0
    state['best_valid_model_acc'] = 0
    state['best_valid_model_f1'] = 0
    state['best_valid_model_idx'] = 0
    net = OrderedDict()

    # weights = pickle.load(open(os.path.join(model_save_dir, weights_outfile), "rb"))

    # spatial stream
    tmp = np.mean(weights['conv1'], axis=1)  # take the average over the 3 RGB channels
    net['model.module.spatial_stream.layer_dict.conv_1.weight'] = torch.from_numpy(tmp[:, np.newaxis, :, :])
    for i in range(2, 6):
        net['model.module.spatial_stream.layer_dict.conv_{}.weight'.format(i)] = torch.from_numpy(
            weights['conv{}'.format(i)])
    for i in range(1, 3):
        net['model.module.spatial_stream.layer_dict.fc_{}.weight'.format(i)] = torch.from_numpy(
            weights['fc{}'.format(i + 5)])
    net['model.module.spatial_stream.layer_dict.logits.weight'.format(i)] = torch.from_numpy(weights['fc8'])

    for i in range(1, 6):
        net['model.module.spatial_stream.layer_dict.conv_{}.bias'.format(i)] = torch.from_numpy(
            weights['conv{}.bias'.format(i)])
    for i in range(1, 3):
        net['model.module.spatial_stream.layer_dict.fc_{}.bias'.format(i)] = torch.from_numpy(
            weights['fc{}.bias'.format(i + 5)])
    net['model.module.spatial_stream.layer_dict.logits.bias'] = torch.from_numpy(weights['fc8.bias'])

    # temporal stream
    expanded = np.tile(weights['conv1'], (1, 57, 1, 1))[:, :170, :, :]
    noise = np.random.normal(loc=np.mean(weights['conv1']), scale=np.std(weights['conv1']) / 2, size=expanded.shape)
    net['model.module.temporal_stream.layer_dict.conv_1.weight'] = torch.from_numpy(expanded + noise)
    for i in range(2, 6):
        net['model.module.temporal_stream.layer_dict.conv_{}.weight'.format(i)] = torch.from_numpy(
            weights['conv{}'.format(i)])
    for i in range(1, 3):
        net['model.module.temporal_stream.layer_dict.fc_{}.weight'.format(i)] = torch.from_numpy(
            weights['fc{}'.format(i + 5)])
    net['model.module.temporal_stream.layer_dict.logits.weight'.format(i)] = torch.from_numpy(weights['fc8'])

    for i in range(1, 6):
        net['model.module.temporal_stream.layer_dict.conv_{}.bias'.format(i)] = torch.from_numpy(
            weights['conv{}.bias'.format(i)])
    for i in range(1, 3):
        net['model.module.temporal_stream.layer_dict.fc_{}.bias'.format(i)] = torch.from_numpy(
            weights['fc{}.bias'.format(i + 5)])
    net['model.module.temporal_stream.layer_dict.logits.bias'] = torch.from_numpy(weights['fc8.bias'])

    state['network'] = net
    return state

# Get Matlab file from http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-2048.mat
if __name__ == '__main__':
    model_save_dir = os.path.join('data','weights')
    # outfilepath = os.path.join(model_save_dir, "{}_{}.pth".format(model_init_name, "pretrained"))
    model_init_name = 'init_model'
    outfilepath = os.path.join(model_save_dir, "{}_{}.pth".format(model_init_name, "pretrained"))

    matlab_file = os.path.join(model_save_dir, 'imagenet-vgg-m-2048.mat')

    # weights_file = 'weights_CNN-M-2048.p'
    # weights_filepath = os.path.join(model_save_dir, weights_file)

    # Extract weights from matlab file:
    print("Extracting weights from matlab file ...")
    weights = extract_weights_from_matlab(matlab_file)

    print("Retrieving init state ...")
    init_state = get_init_state(weights)

    print("Created network:")
    for i, s in enumerate(init_state['network']):
        #     print(i, s)
        print(init_state['network'][s].shape)

    print("Saving weights to {} ...".format(outfilepath))
    # save state as initial state that includes the pre-trained weights from ImageNet
    torch.save(init_state, f=outfilepath)

    print("Done")