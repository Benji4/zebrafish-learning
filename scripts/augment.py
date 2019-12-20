import numpy as np
import time
import cv2
import os
import h5py
from multiprocessing import Process
from PIL import Image
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

coordinates = [8,16,24]
offsets_lookup = np.array([(i,j) for i in coordinates for j in coordinates])

def get_random_masks(size, num_frames, num_blocks, num_frames_per_block, remainder=0):
    n, b, k = num_frames, num_blocks, num_frames_per_block
    masks = np.zeros((size, b * k + remainder), dtype=np.bool_)
    masks[:,
    :b * k:k] = True  # set every kth element to 1, because each block needs at least one frame (here called fixed frame)

    for j in range(size):  # for each mask do
        # From the remaining available frames, make a random selection:
        a = np.arange(b * (k - 1) + remainder)  # array which excludes the first element of each block
        idx = np.random.choice(a, size=(n - b), replace=False)  # b of n frames have already been distributed
        idx += np.array(idx / (k - 1), dtype=np.uint8) + 1  # get the index into masks
        idx[idx > b * k] -= 1  # the frames which do not belong to a block have no fixed frame in their block
        masks[j, idx] = True  # set these elements to True too

        # Random permutation within each block:
        for i in range(b):
            idx = i * k
            masks[j, idx:idx + k] = np.random.permutation(masks[j, idx:idx + k])

    return masks

def augment_data(X, X_flow, is_flipped, offset_idxs):
    # start = time.time()
    X      = flip_vertical(X, is_flipped, True)
    X_flow = flip_vertical(X_flow, is_flipped, False)
    X      = crop(X, offset_idxs, True)
    X_flow = crop(X_flow, offset_idxs, False)
    # print("Augmenting current batch took", time.time()-start,"seconds.")
    return X, X_flow

# works on original data, i.e. (b,150,256,256) and flow data, i.e. (b, 85, 256, 256, 2)
def flip_vertical(X, is_flipped, spatial):
    flip_idxs = np.nonzero(is_flipped)
    if spatial: ax = 1
    else: ax = 2
    X[flip_idxs] = np.flip(X[flip_idxs], axis=ax)
    return X

# works on original data, i.e. (b,256,256) and flow data, i.e. (b, 85, 256, 256, 2)
def crop(X, offset_idxs, spatial):
    offsets = offsets_lookup[offset_idxs]
    X_cropped = []
    if spatial:
        for i, offset in enumerate(offsets):
            X_cropped.append(X[i,offset[1] : offset[1] + 224, offset[0] : offset[0] + 224])
    else:
        for i, offset in enumerate(offsets):
            X_cropped.append(X[i,:,offset[1] : offset[1] + 224, offset[0] : offset[0] + 224])

    X_cropped = np.array(X_cropped)
    return X_cropped

# one mask per sample in batch X (X and masks must have same first dimension)
# works only on original data, i.e. (b,150,256,256)
def subsample(X, masks):
    # Apply masks to X
    X_subsampled = X[masks].reshape((X.shape[0], 86, X.shape[2], X.shape[3]))
    return X_subsampled

def get_farneback_flow(prvs, next): # as float32
    return np.float32(cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.8, levels=10, winsize=10, iterations=10, poly_n=13, poly_sigma=1.8, flags=0))

def get_flow_from_video(video):  # e.g. (86,256,256) returns (85,256,256,2)
    # start = time.time()
    video_flow = np.empty((video.shape[0] - 1, video.shape[1], video.shape[2], 2), dtype=np.float32)
    for i in range(video.shape[0] - 1):
        prvs = video[i]
        next = video[i + 1]
        video_flow[i] = get_farneback_flow(prvs, next)

    # print("Elapsed time: %.2fs" % (time.time()-start))
    return video_flow


def compress_frame(frame):
    min, max = np.min(frame), np.max(frame)
    frame = cv2.normalize(frame,None,0,255,cv2.NORM_MINMAX)
    out = BytesIO()
    im = Image.fromarray(frame, "F")
    im = im.convert('RGB')
    im.save(out,format='JPEG',quality=40) # use jpeg compression
    compr = Image.open(out)
    gray = np.array(compr)[:,:,0] # channels 0, 1, and 2 are all equal, because it is grayscale
    return gray, min, max

def compress_batch(batch): # batch of optical flow
    # start = time.time()
    minmax = np.empty((len(batch),85,2,2), dtype=np.float32)
    for i,sample in enumerate(batch):
        for j, flow_frame in enumerate(sample):
            for k in range(2):  # for x and y flow
                flow_frame[:, :, k], min, max = compress_frame(flow_frame[:, :, k])
                minmax[i, j, 0, k] = min
                minmax[i, j, 1, k] = max


    batch = np.uint8(batch)
    # print("Batch compression took", time.time()-start,"seconds.")
    return batch, minmax

def run_worker(worker_id, X_segment, y_segment):
    np.random.seed(462019 + worker_id) # set unique seed for each worker process to use unique set of masks with each worker

    # print("X_segment.shape", X_segment.shape)

    worker_segment = X_segment.shape[0] # worker_segment == segment_size, except for the final batch

    total = num_masks_per_worker * worker_segment  # number of resulting masks 47 masks * 100 samples
    f = 86  # number of resulting frames (succeeding flow calc brings it to 85)
    b = 75  # number of blocks
    k = 2  # number of frames per block
    remainder = 0  # number of frames that don't make up a whole block, e.g. 149 % 2 = 1 or 149 % 3 = 2
    masks = get_random_masks(size=total, num_frames=f, num_blocks=b, num_frames_per_block=k, remainder=remainder)
    masks = masks.reshape((num_masks_per_worker, worker_segment, 150))
    # print("masks.shape", masks.shape)

    # compr = 3

    n = worker_segment * num_masks_per_worker * 18  # the hdf5 file will store batch|batch|batch|batch|...

    # allocate space for datasets
    hdf5_file = '{0}/worker-{1}.hdf5'.format(data_folder, worker_id)
    if os.path.exists(hdf5_file):
        os.remove(hdf5_file)
    with h5py.File(hdf5_file, 'w') as file:
        # pre allocate space for this worker's file; lzf compression is fast:
        file.create_dataset("inputs_orig", (n, 224, 224), dtype=np.uint8, compression="gzip", compression_opts=9)
        file.create_dataset("inputs_flow", (n, 170, 224, 224), dtype=np.uint8, compression="gzip", compression_opts=9)
        file.create_dataset("targets", (n,), dtype=np.bool_, compression="gzip", compression_opts=9)
        file.create_dataset("minmax", (n, 170, 2), dtype=np.float32, compression="gzip", compression_opts=9)

    # print("Worker {0} elapsed time: {1:.2f} secs".format(worker_id, time.time() - start))
    print("Worker {0:2} applying {1:2} masks, computing flow, and augmenting ...".format(worker_id, masks.shape[0]))
    for i in range(masks.shape[0]):

        # apply mask and compute flow
        X_flow = np.empty((worker_segment, 85, 256, 256, 2), dtype=np.float32)
        X_sub = subsample(X_segment, masks[i])
        for j in range(masks.shape[1]):
            X_flow[j] = get_flow_from_video(X_sub[j])

        frame_idxs = np.random.randint(0, 86, worker_segment) # we sample one frame from the subsampled frames
        X_spatial = X_sub[
            np.arange(worker_segment), frame_idxs]  # keep only one frame per sample for the spatial stream

        # then apply flip and crop to X_flow:
        # Compute augmentation lookup (for flipping and cropping)
        aug = np.arange(18).repeat(worker_segment).reshape((18, worker_segment))
        for ll in range(worker_segment):
            np.random.shuffle(aug[:, ll])  # in-place

        max_number = worker_segment  # just for debugging

        # print("Worker {0:2} elapsed time: {1:.2f} secs".format(worker_id, time.time() - start))

        # print("Worker {0:2} saving 18 batches to hdf5 file {1} ...".format(worker_id, hdf5_file))

        # go through all combinations of flipping and cropping
        for k in range(18):
            is_flipped, crop_ids = aug[k] % 2, aug[k] // 2
            batch, flow_batch = augment_data(X=X_spatial[:max_number], X_flow=X_flow[:max_number],
                                             is_flipped=is_flipped[:max_number], offset_idxs=crop_ids[:max_number])
            targets = y_segment[:max_number]
            # we never changed the order of the original batch X_segment, thus y_segment applies here just like before.

            compressed_flow_batch, minmax = compress_batch(flow_batch)

            # Put x and y frames alternating as channels:
            compressed_flow_batch = compressed_flow_batch.swapaxes(3, 4).swapaxes(2, 3).reshape(worker_segment, 170, 224,224)
            minmax = minmax.swapaxes(2, 3).reshape(worker_segment, 170, 2)

            # print("compressed_flow_batch.shape",compressed_flow_batch.shape)

            # save batch to uncompressed file
            # outpath = str(batches_folder) + "/batch_" + str(worker_id) + "_" + str(i) + "_" + str(k)
            # print("Saving batch to uncompressed file in", outpath)
            # np.savez(outpath, inputs_orig=batch, inputs_flow=compressed_flow_batch,
            #                     targets=y_segment[:max_number])


            idx = worker_segment * (
                        k + i * 18)  # i is 18 batches, k is 1 batch, worker_segment is the size of one batch
            end = idx + worker_segment

            with h5py.File(hdf5_file, 'a') as file: # open file in append mode and close afterwards
                dset_inputs_orig = file['inputs_orig']
                dset_inputs_flow = file['inputs_flow']
                dset_targets = file['targets']
                dset_minmax = file['minmax']

                dset_inputs_orig[idx:end] = batch
                dset_inputs_flow[idx:end] = compressed_flow_batch
                dset_targets[idx:end] = targets
                dset_minmax[idx:end] = minmax

        print("Worker {0:2}, elapsed time: {1:.2f} secs, masks done: {2:2}/{3:2}".format(worker_id, time.time() - start, i+1, masks.shape[0]))

    print("Worker {0:2} done within {1:.2f} secs".format(worker_id, time.time() - start))
    return






start = time.time()

# data_folder = '/afs/inf.ed.ac.uk/group/project/s1832591/final'
data_folder = '/disk/scratch/14f'
filepath = '{}/zebrafish_all_cut_agarose_85.npz'.format(data_folder) # all the original data, i.e. events and targets (segment should have 100 original samples)
final_output = '{}/all.hdf5'.format(data_folder)
num_workers = 38
num_masks_per_worker = 8
segment_size = 32
num_workers_per_segment = 1


if __name__ == '__main__':
    np.random.seed(462019)

    print("Loading data ...")
    loaded = np.load(filepath)  # entire original data
    X, y = loaded['inputs'], loaded['targets']
    print("Loading data done in {0:.2f} secs".format(time.time()-start))

    jobs = []
    for worker_id in range(num_workers):
        # Slice out segment for worker:
        idx = (worker_id // num_workers_per_segment) * segment_size
        end = idx + segment_size
        print("Worker {0:2} loading segment {1:4} - {2:4} ...".format(worker_id, idx, end))
        X_segment, y_segment = X[idx:end], y[idx:end]  # slicing returns an empty array if idx or end are out of bounds

        # Safety check:
        if X_segment.size == 0 or y_segment.size == 0:
            print("WARNING: Worker {} has received an empty segment {}-{} !".format(worker_id, idx, end))
            continue

        p = Process(target=run_worker, args=(worker_id, X_segment, y_segment)) # hand over reference to data
        jobs.append(p)
        p.start()

    # Wait for all processes to be done:
    for p in jobs:
        p.join()

    print("######## All processes done! ########")


    # Create one file linking to all other files once all processes are done:
    print("Creating final output file in {} ...".format(final_output))
    if os.path.exists(final_output):
        os.remove(final_output)
    with h5py.File(final_output, 'w') as f:
        for i in range(num_workers):
            # external links should be relative to the parent hdf5-file in path final_ouput
            f['worker-{}-inputs_orig'.format(i)] = h5py.ExternalLink('worker-{}.hdf5'.format(i), "inputs_orig")
            f['worker-{}-inputs_flow'.format(i)] = h5py.ExternalLink('worker-{}.hdf5'.format(i), "inputs_flow")
            f['worker-{}-targets'.format(i)] = h5py.ExternalLink('worker-{}.hdf5'.format(i), "targets")
            f['worker-{}-minmax'.format(i)] = h5py.ExternalLink('worker-{}.hdf5'.format(i), "minmax")

    print("Total duration: {0:.2f}".format(time.time()-start))




