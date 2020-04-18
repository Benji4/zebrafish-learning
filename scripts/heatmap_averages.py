import numpy as np
import h5py
import os

if __name__ == '__main__':

    analysis_file = '/disk/scratch/analysis/analysis.hdf5' # TODO path to file
    orig_file = '/disk/scratch/9f/all.hdf5' # TODO path to file
    out_dir = 'Downloads' # TODO path to file
    worker_ids = [32, 33, 34, 35, 36, 37]
    n = 1842

    print("Loading data ...")
    with h5py.File(analysis_file, 'r') as f:
        analysis_s = f['spatial'][:]
        analysis_t = f['temporal'][:]
        kept_idxs = f['kept_idxs'][:]
    analysis_s = analysis_s[:n]
    analysis_t = analysis_t[:n]
    kept_idxs = kept_idxs[:n]

    y = []
    with h5py.File(orig_file, 'r') as f:
        for i, worker_id in enumerate(worker_ids):
            tmp = np.asarray(f['worker-{}-targets'.format(worker_id)][:], dtype=np.int64)
            if i == 0: y = tmp
            else: y = np.concatenate((y, tmp))
    y = np.array(y).flatten()
    y = y[kept_idxs]
    print("Number of samples: {}".format(len(y)))

    print("Calculating averages ...")
    positives = np.nonzero(y == 1)
    negatives = np.nonzero(y == 0)
    avg_over_samples = np.mean(analysis_t, axis=0)
    avg_over_frames = np.mean(analysis_t, axis=1)
    avg_over_positive_samples = np.mean(analysis_t[positives], axis=0)
    avg_over_negative_samples = np.mean(analysis_t[negatives], axis=0)

    avg_over_samples_s = np.mean(analysis_s, axis=0)
    avg_over_frames_s = np.mean(analysis_s, axis=1)
    avg_over_positive_samples_s = np.mean(analysis_s[positives], axis=0)
    avg_over_negative_samples_s = np.mean(analysis_s[negatives], axis=0)

    print("avg_over_samples {}".format(avg_over_samples.shape))
    print("avg_over_frames {}".format(avg_over_frames.shape))
    print("avg_over_positive_samples {}".format(avg_over_positive_samples.shape))
    print("avg_over_negative_samples {}".format(avg_over_negative_samples.shape))
    print("avg_over_samples_s {}".format(avg_over_samples_s.shape))
    print("avg_over_frames_s {}".format(avg_over_frames_s.shape))
    print("avg_over_positive_samples_s {}".format(avg_over_positive_samples_s.shape))
    print("avg_over_negative_samples_s {}".format(avg_over_negative_samples_s.shape))

    np.savez(os.path.join(out_dir, 'avgs.npz'),
             avg_over_samples=avg_over_samples, avg_over_frames=avg_over_frames,
             avg_over_positive_samples=avg_over_positive_samples, avg_over_negative_samples=avg_over_negative_samples,
             avg_over_samples_s=avg_over_samples_s, avg_over_frames_s=avg_over_frames_s,
             avg_over_positive_samples_s=avg_over_positive_samples_s, avg_over_negative_samples_s=avg_over_negative_samples_s,
             y=y)


    print("Done")



