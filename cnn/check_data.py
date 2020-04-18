import numpy as np
import h5py
import time

if __name__ == '__main__':

    num_workers = 38
    segment_size = 32
    num_masks_per_worker = 8

    flipping_factor = 2
    cropping_factor = 9
    aug_mult_factor_per_worker = num_masks_per_worker * flipping_factor * cropping_factor # masking, flipping, cropping
    num_samples_per_worker = segment_size * aug_mult_factor_per_worker # used to find the right file in Dataset-class

    all_file = '/disk/scratch/12f/all.hdf5' # TODO path to file

    problem_ids = []

    start = time.time()

    print("Outputting the average of y for each worker in {} ...".format(all_file))
    with h5py.File(all_file, 'r') as f:
        for i in range(num_workers):
            print("Worker {}".format(i))
            shape = f['worker-{}-targets'.format(i)].shape
            ys = f['worker-{}-targets'.format(i)][:]
            # print("Elapsed time: {}".format(time.time() - start))
            print(np.average(ys))
            print("")


    print("Ouputting key numbers (averages and shapes) via {} ...".format(all_file))
    with h5py.File(all_file, 'r') as f:
        for i in range(num_workers):
            try:
                print("Worker {}".format(i))
                print(np.average(f['worker-{}-inputs_orig'.format(i)][-1]))
                print(np.average(f['worker-{}-inputs_flow'.format(i)][-1]))
                print(np.average(f['worker-{}-targets'.format(i)][-1]))
                print(np.average(f['worker-{}-minmax'.format(i)][-1]))

                print(f['worker-{}-inputs_orig'.format(i)].shape)
                print(f['worker-{}-inputs_flow'.format(i)].shape)
                print(f['worker-{}-targets'.format(i)].shape)
                print(f['worker-{}-minmax'.format(i)].shape)
                print("Elapsed time: {}".format(time.time() - start))
                print("")

            except:
                print("Problem with worker {} !".format(i))
                problem_ids.append(i)


    print("problem_ids: {}".format(problem_ids))

    problem_ids2 = []
    print("Checking all samples for accessbility ...")
    with h5py.File(all_file, 'r') as f:
        for i in range(num_workers):
            print("Checking worker {}".format(i))
            for j in range(num_samples_per_worker):
                try:
                    x = f['worker-{}-inputs_orig'.format(i)][j]
                    x = f['worker-{}-inputs_flow'.format(i)][j]
                    x = f['worker-{}-targets'.format(i)][j]
                    x = f['worker-{}-minmax'.format(i)][j]
                except:
                    print("Problem with worker {} !".format(i))
                    problem_ids2.append(i)

    print("problem_ids: {}; problem_ids2: {}".format(problem_ids, problem_ids2))

    print("Done")