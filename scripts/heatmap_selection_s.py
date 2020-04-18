import numpy as np
import matplotlib.pyplot as plt
import innvestigate.utils.visualizations as ivis
import cv2
import h5py
import os

#######  Outputs a selection of spatial heatmaps  #######

def reverse_rescale(X, min, max):
    """ in-place for minimal RAM consumption """
    # print(X.shape, min.shape, max.shape)
    #     X = min + X * (max-min) / 255
    max -= min
    max /= 255
#     print(X.shape, min.shape, max.shape)
    X *= max[...,np.newaxis,np.newaxis]  # add two empty axes for broadcasting
    X += min[...,np.newaxis,np.newaxis]
    return X

def get_imshow_heatmap(heat):
    h = np.tile(heat[np.newaxis, np.newaxis], (1, 3, 1, 1))
    h = h.swapaxes(1,2).swapaxes(2,3)
    h = ivis.heatmap(h)[0]
    return h

if __name__ == '__main__':
    np.random.seed(462019)

    # analysis_file = 'data/analysis.hdf5'
    # orig_file = 'data/ana.hdf5'
    analysis_file = '/disk/scratch/analysis/analysis.hdf5' # TODO path to file
    orig_file = '/disk/scratch/9f/all.hdf5' # TODO path to file
    out_dir = 'Downloads' # TODO path to file


    alpha_s = 0.75
    alpha_t = 0.8

    num_samples_per_worker = 576

    names = ["TP", "TN", "FP", "FN", "TP", "TP", "TN", "TN"]
    original = np.array([676, 1470, 1891, 437, 2913, 2339,1031, 3030]) # index into original set of 3420 samples
    analysis = np.array([395, 622, 744, 260, 1376, 1026,554, 1483]) # index into analysis set of 1842 samples

    worker_ids = original // num_samples_per_worker + 32 # test set starting at worker 32
    sample_idxs_in_worker = original % num_samples_per_worker

    print("Worker ids: {}".format(worker_ids))
    print("sample_idxs_in_worker: {}".format(sample_idxs_in_worker))

    fig = plt.figure(figsize=(8.8,5))

    for k in range(len(names)): # iterate over the samples to plot


        name = names[k]
        analysis_idx = analysis[k]
        worker_id = worker_ids[k]
        sample_idx_in_worker = sample_idxs_in_worker[k]

        print("Loading sample {} ...".format(names[k]))
        with h5py.File(analysis_file, 'r') as f:
            analysis_s = f['spatial'][analysis_idx]
        with h5py.File(orig_file, 'r') as f:
            x = f['worker-{}-inputs_orig'.format(worker_id)][sample_idx_in_worker]


        ################# SPATIAL ####################

        print("Generating spatial heatmap ...")

        plt.subplot(2,4,k+1)

        # print(analysis.shape)
        #
        x = np.tile(x[:,:,np.newaxis], (1, 1, 3))
        plt.imshow(x)

        h = get_imshow_heatmap(analysis_s)
        plt.imshow(h, alpha=alpha_s)
        plt.tick_params(
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False
        )
        plt.title(name)


    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spatial_selection.pdf"), bbox_inches='tight')

    print("Done")



