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
#     print(X.shape, min.shape, max.shape)
    X *= max[...,np.newaxis,np.newaxis]  # add two empty axes for broadcasting
    X += min[...,np.newaxis,np.newaxis]
    return X

if __name__ == '__main__':
    np.random.seed(462019)

    analysis_file = '/disk/scratch/analysis/analysis.hdf5'
    orig_file = '/disk/scratch/9f/all.hdf5'
    out_dir = 'Downloads'


    alpha_s = 0.75
    alpha_t = 0.8

    num_samples_per_worker = 576

    names = ["tp"]
    original = np.array([676]) # index into original set of 3420 samples
    analysis = np.array([395]) # index into analysis set of 1842 samples
    # names = names[-5:]
    # original = original[-5:]
    # analysis = analysis[-5:]

    # Other experiment:
    # original = np.array([1955, 2179, 1276, 2051, 2019, 2083, 2275, 2115, 1763, 1987,
    #    1731, 1859, 2157, 2093, 2375, 2870, 2358, 1564, 2061, 1997, 2535,
    #    2285, 1933, 2631, 2599, 2823, 1827, 2243, 2695, 2422, 1923, 2550,
    #    2454, 2774, 2646, 2614])
    # analysis = np.array([785,922,612,844,821,865,983,887,659,801,639,725,912,874,1048,1340,1036,626,852,808,1135,989,775,1196,1178,1310,704,964,1233,1072,765,1144,1088,1281,1205,1186])
    # names = ["fp_check_{}".format(i) for i in range(len(analysis))]
    # original = original[18:21]
    # analysis = analysis[18:21]
    # names = names[18:21]

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


        ################# TEMPORAL ####################

        print("Generating temporal heatmap ...")

        num_rows = int(85 / 5)  # 17
        a = 0
        b = a + 5 * num_rows

        xx = x_flow.copy()
        xx = reverse_rescale(xx.astype(np.float32), minmax[:, 0], minmax[:, 1])
        xx = xx.reshape(85, 2, 224, 224).swapaxes(1, 2).swapaxes(2, 3)
        # print(x_flow.shape)
        flow = xx[41]

        fig = plt.figure(figsize=(3.2, 3))

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # get magnitude and direction/angle
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = 255  # sets hue
        hsv[..., 1] = 255  # # Sets image saturation to maximum
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # sets value/brightness
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        im = plt.imshow(255-gray, cmap='gray')

        plt.tick_params(left=False, labelleft=False, right=False, labelright=False, bottom=False, labelbottom=False)

        plt.title('Single Frame', fontsize=12)
        plt.ylabel('Temporal', fontsize=12)

        analyzed_frame = analysis_t[np.newaxis, :, :]
        a = np.tile(analyzed_frame, (1, 3, 1, 1))
        a = a.swapaxes(1, 2).swapaxes(2, 3)
        h = ivis.heatmap(a)
        h = h[0]

        plt.imshow(h, alpha=alpha_t)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "original.pdf"), bbox_inches='tight')
        plt.clf()


    print("Done")



