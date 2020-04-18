import numpy as np
import matplotlib.pyplot as plt
import innvestigate.utils.visualizations as ivis
import cv2
import h5py
import os

def reverse_rescale(X, min, max):
    """ in-place for minimal RAM consumption """
    # print(X.shape, min.shape, max.shape)
    #     X = min + X * (max-min) / 255
    max -= min
    max /= 255
    X *= max[...,np.newaxis,np.newaxis]  # add two empty axes for broadcasting
    X += min[...,np.newaxis,np.newaxis]
    return X

if __name__ == '__main__':
    np.random.seed(462019)

    analysis_file = '/disk/scratch/analysis/analysis.hdf5' # TODO path to file
    orig_file = '/disk/scratch/9f/all.hdf5' # TODO path to file
    out_dir = 'Downloads' # TODO path to file


    alpha_s = 0.75
    alpha_t = 0.8

    num_samples_per_worker = 576

    # We found these indices via a confidence analysis as shown in evaluate_probs.ipynb
    names = ["tp", "tn", "fp", "fn", "tp_1","tp_2","tp_3", "tp_4", "tp_5", "tn_1", "tn_2", "tn_3", "tn_4", "tn_5"]
    original = np.array([676, 1470, 1891, 437, 2913, 810, 2339, 3389, 1809, 3330,2912,3263,310,2068]) # index into original set of 3420 samples
    analysis = np.array([395, 622, 744, 260, 1376, 450, 1026, 1814, 692, 1758,1375,1698,187,856]) # index into analysis set of 1842 samples
    # The analysis set is only a subset of the original set, because the iNNvestigate analysis processes some samples faultily,
    # which we did not keep for further analysis.


    worker_ids = original // num_samples_per_worker + 32 # test set starting at worker 32
    sample_idxs_in_worker = original % num_samples_per_worker

    print("Worker ids: {}".format(worker_ids))
    print("sample_idxs_in_worker: {}".format(sample_idxs_in_worker))

    for k in range(len(names)): # iterate over the samples to plot

        analysis_idx = analysis[k]
        worker_id = worker_ids[k]
        sample_idx_in_worker = sample_idxs_in_worker[k]

        print("Loading sample {} ...".format(names[k]))
        with h5py.File(analysis_file, 'r') as f:
            analysis_s = f['spatial'][analysis_idx]
            analysis_t = f['temporal'][analysis_idx]
        with h5py.File(orig_file, 'r') as f:
            x = f['worker-{}-inputs_orig'.format(worker_id)][sample_idx_in_worker]
            x_flow = f['worker-{}-inputs_flow'.format(worker_id)][sample_idx_in_worker]
            minmax = f['worker-{}-minmax'.format(worker_id)][sample_idx_in_worker]


        ################# SPATIAL ####################

        print("Generating spatial heatmap ...")

        fig = plt.figure()

        # print(analysis.shape)
        #
        x = np.tile(x[:,:,np.newaxis], (1, 1, 3))
        plt.imshow(x)

        a = np.tile(analysis_s, (1, 3, 1, 1))
        a = a.swapaxes(1,2).swapaxes(2,3)
        # print(a.shape)
        h = ivis.heatmap(a)
        plt.imshow(h[0], alpha=alpha_s)
        plt.tick_params(
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "h_spatial_{}.pdf".format(names[k])), bbox_inches='tight')
        plt.clf()


        ################# TEMPORAL ####################

        print("Generating temporal heatmap ...")

        num_rows = int(85 / 5)  # 17
        a = 0
        b = a + 5 * num_rows

        xx = x_flow.copy()
        xx = reverse_rescale(xx.astype(np.float32), minmax[:, 0], minmax[:, 1])
        xx = xx.reshape(85, 2, 224, 224).swapaxes(1, 2).swapaxes(2, 3)
        # print(x_flow.shape)
        subset = xx[a:b]

        fig = plt.figure(figsize=(15, num_rows * 3))
        for i, flow in enumerate(subset):
            plt.subplot(num_rows, 5, i + 1)

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
            plt.title(i)

            analyzed_frame = analysis_t[np.newaxis, i, :, :]
            a = np.tile(analyzed_frame, (1, 3, 1, 1))
            a = a.swapaxes(1, 2).swapaxes(2, 3)
            h = ivis.heatmap(a)
            h = h[0]

            plt.imshow(h, alpha=alpha_t)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "h_temporal_{}.pdf".format(names[k])), bbox_inches='tight')
        plt.clf()


    print("Done")



