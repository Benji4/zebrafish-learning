import numpy as np
import torch
import sys
from data_providers import Dataset
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from model_architectures import TwoStreamNetwork
from torch.utils import data

if __name__ == '__main__':
    debug = False

    args = get_args()  # get arguments from command line


    experiment_name = args.experiment_name
    num_epochs = args.num_epochs

    batch_size = args.batch_size

    num_orig_samples = 1214
    segment_size = 32 # used by DataLoader for internally finding the right hdf5 file and idx to an ID
    num_workers_per_segment = 1

    if debug:
        args.set = '9f'

    if args.set == '9f' or args.set == '13f':
        num_masks_per_worker = 1
    elif args.set == '10f':
        num_masks_per_worker = 4
    elif args.set == '11f':
        num_masks_per_worker = 32
    elif args.set == '12f' or args.set == '14f':
        num_masks_per_worker = 8
    else:
        sys.exit("ERROR: Unknown set '{}'".format(args.set))

    seed = 462019
    num_workers = 32 # for data loading from disk
    all_file = 'batches_{}/all.hdf5'.format(args.set)

    if debug:
        num_orig_samples = 9
        segment_size = 3
        num_workers_per_segment = 1
        num_masks_per_worker = 1
        num_workers = 1
        all_file = '../batches_data_testing/all.hdf5'
        batch_size = 3
        num_epochs = 1
        args.pretrained = False


    use_gpu = True
    continue_from_epoch = args.continue_from_epoch
    gpu_id = "0,1,2,3"
    weight_decay_coefficient = args.weight_decay_coefficient

    dataset = "zebrafish"
    num_flow_channels = 170 # 85 * 2 = 170
    image_height = 224
    image_width = image_height # we assume equal height and width

    flipping_factor = 2
    cropping_factor = 9
    aug_mult_factor_per_worker = num_masks_per_worker * flipping_factor * cropping_factor # masking, flipping, cropping
    aug_mult_factor = num_workers_per_segment * aug_mult_factor_per_worker # - by how much is one original sample augmented in total
    num_samples_per_segment = segment_size * aug_mult_factor # used for train, valid, test split
    num_samples_per_worker = segment_size * aug_mult_factor_per_worker # used to find the right file in Dataset-class
    # num_samples_per_segment != num_samples_per_worker iff num_workers_per_segment != 1

    # Train, valid, test split by segment first, this is to ensure that the split is done on the original samples
    # such that one original video cannot be in both train and test or train and valid or valid and test:
    num_segments = num_orig_samples // segment_size
    num_segments_train = int(num_segments * 0.77)
    num_segments_valid = int(num_segments * 0.13)
    # num_segments_test = num_segments - (num_segments_train + num_segments_valid) # test gets the 14 samples which did not fit in any other segment

    num_batches = args.num_batches

    print("experiment_name={0}, num_epochs={1}, batch_size={2}, num_workers={3}, all_file={4}, continue_from_epoch={5}, num_batches={6}, weight_decay_coefficient={7}, lr={8}, schedule={9}".format(experiment_name, num_epochs, batch_size, num_workers, all_file, continue_from_epoch, num_batches, weight_decay_coefficient, args.lr, args.schedule))


    if debug:
        num_segments_train = 1
        num_segments_valid = 1

    # Then calculate the indices on the augmented data:
    break_one = num_segments_train * num_samples_per_segment
    break_two = break_one + num_segments_valid * num_samples_per_segment
    num_samples = num_orig_samples * aug_mult_factor
    train_range = range(0,         break_one)
    valid_range = range(break_one, break_two)
    test_range  = range(break_two, num_samples) # num_samples should be 1214 * 94 * 2 * 9 = 2,054,088

    print("train_range, valid_range, test_range: {0}, {1}, {2}".format(train_range, valid_range, test_range))
    
    rng = np.random.RandomState(seed=seed)  # set the seeds for the experiment
    torch.manual_seed(seed=seed)  # sets pytorch's seed
    np.random.seed(seed)

    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': num_workers}

    partition = {'train': train_range, 'valid': valid_range, 'test': test_range}

    # Generators
    print("Creating data providers...")
    train_dataset = Dataset(partition['train'], all_file, num_samples_per_worker)
    train_data = data.DataLoader(train_dataset, **params)

    val_dataset = Dataset(partition['valid'], all_file, num_samples_per_worker)
    val_data = data.DataLoader(val_dataset, **params)

    test_dataset = Dataset(partition['test'], all_file, num_samples_per_worker)
    test_data = data.DataLoader(test_dataset, **params)

    num_output_classes = 2

    # data_store = data_providers.DataStore()

    custom_net = TwoStreamNetwork(
        input_shape=(batch_size, num_flow_channels, image_height, image_width), dropout_rate=args.dropout_rate)

    experiment = ExperimentBuilder(network_model=custom_net,
                                   experiment_name=experiment_name,
                                   num_epochs=num_epochs,
                                   weight_decay_coefficient=weight_decay_coefficient,
                                   lr = args.lr,
                                   gpu_id=gpu_id, use_gpu=use_gpu,
                                   continue_from_epoch=continue_from_epoch,
                                   num_batches=num_batches,
                                   train_data=train_data, valid_data=val_data,
                                   test_data=test_data,
                                   pretrained=args.pretrained,
                                   schedule=args.schedule)  # build an experiment object

    if not args.only_output_results:
        total_losses, test_losses = experiment.run_experiment()  # run experiment and return experiment metrics
        print(total_losses, test_losses)
    else:
        # valid_losses = experiment.output_summary(type='valid', epoch_idx=continue_from_epoch)
        test_losses = experiment.output_summary(type='test', epoch_idx=continue_from_epoch)


    print("Done")