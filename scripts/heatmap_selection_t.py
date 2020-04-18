import numpy as np
import matplotlib.pyplot as plt
import innvestigate.utils.visualizations as ivis
import cv2
import h5py
import os

######  Outputs a selection of temporal heatmaps  ######

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

    names = ["TP","TP", "TP", "TN", "TN"]
    original = np.array([2913, 810, 1809, 3122, 1031]) # index into original set of 3420 samples
    analysis = np.array([1376, 450, 692, 1571, 554]) # index into analysis set of 1842 samples
    frame_idxs = np.array([17,19,70, 33, 14])

    worker_ids = original // num_samples_per_worker + 32 # test set starting at worker 32
    sample_idxs_in_worker = original % num_samples_per_worker

    print("Worker ids: {}".format(worker_ids))
    print("sample_idxs_in_worker: {}".format(sample_idxs_in_worker))

    fig = plt.figure(figsize=(8.8, 6))

    for k in range(len(names)): # iterate over the samples to plot


        name = names[k]
        analysis_idx = analysis[k]
        worker_id = worker_ids[k]
        sample_idx_in_worker = sample_idxs_in_worker[k]
        frame_idx = frame_idxs[k]

        print("Loading sample {} ...".format(names[k]))
        with h5py.File(analysis_file, 'r') as f:
            analysis_t = f['temporal'][analysis_idx][frame_idx]
        with h5py.File(orig_file, 'r') as f:
            x_flow = f['worker-{}-inputs_flow'.format(worker_id)][sample_idx_in_worker]
            minmax = f['worker-{}-minmax'.format(worker_id)][sample_idx_in_worker]

        # print("analysis_s.shape {}".format(analysis_s.shape))
        # print("analysis_t.shape {}".format(analysis_t.shape))
        # print(np.sum(np.abs(analysis_t[0])))
        # print(np.sum(np.abs(analysis[1])))


        ################# TEMPORAL ####################

        print("Generating temporal heatmap ...")

        plt.subplot(1,5,k+1)

        xx = x_flow.copy()
        xx = reverse_rescale(xx.astype(np.float32), minmax[:, 0], minmax[:, 1])
        xx = xx.reshape(85, 2, 224, 224).swapaxes(1, 2).swapaxes(2, 3)
        # print(x_flow.shape)

        flow = xx[frame_idx]

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # get magnitude and direction/angle
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = 255  # sets hue
        hsv[..., 1] = 255  # # Sets image saturation to maximum
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # sets value/brightness
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        im = plt.imshow(255-gray, cmap='gray')

        plt.tick_params(
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False
        )
        plt.title(name)

        h = get_imshow_heatmap(analysis_t)

        plt.imshow(h, alpha=alpha_t)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "temporal_selection.pdf"), bbox_inches='tight')


    print("Done")



