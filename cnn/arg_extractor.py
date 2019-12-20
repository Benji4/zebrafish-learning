import argparse
import json


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='helper script')

    parser.add_argument('--exp_id', nargs="?", type=int, default=0, help='Give your experiment an id.')
    parser.add_argument('--net_id', nargs="?", type=int, default=0, help='0=QuinnNetwork, 1=ResNet-50.')
    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch_size for experiment')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    parser.add_argument('--dataset_name', type=str, help='Dataset on which the system will train/eval our model')
    parser.add_argument('--seed', nargs="?", type=int, default=462019,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--image_num_channels', nargs="?", type=int, default=1,
                        help='The channel dimensionality of our image-data')
    parser.add_argument('--image_height', nargs="?", type=int, default=28, help='Height of image data')
    parser.add_argument('--image_width', nargs="?", type=int, default=28, help='Width of image data')
    parser.add_argument('--dim_reduction_type', nargs="?", type=str, default='strided_convolution',
                        help='One of [strided_convolution, dilated_convolution, max_pooling, avg_pooling]')
    parser.add_argument('--num_layers', nargs="?", type=int, default=4,
                        help='Number of convolutional layers in the network (excluding '
                             'dimensionality reduction layers)')
    parser.add_argument('--num_filters', nargs="?", type=int, default=64,
                        help='Number of convolutional filters per convolutional layer in the network (excluding '
                             'dimensionality reduction layers)')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=1e-05,
                        help='Weight decay to use for Adam')
    parser.add_argument('--filepath_to_arguments_json_file', nargs="?", type=str, default=None,
                        help='')
    parser.add_argument('--dataset', nargs="?", type=int, default=0, help='Which dataset to use. One of 0,1,2,3.')
    parser.add_argument('--num_batches', nargs="?", type=int, default=float("inf"), help='Limit number of batches for debugging')
    parser.add_argument('--set', nargs="?", type=str, default=None,help='Which set to use for batches, e.g. 10f')
    parser.add_argument('--dropout_rate', nargs="?", type=float, default=0.9,
                        help='dropout_rate for TwoStreamArchitecture')
    parser.add_argument('--lr', nargs="?", type=float, default=1e-03,
                        help='Learning Rate for Adam')
    parser.add_argument('--pretrained', nargs="?", type=str2bool, default=True,
                        help='A flag indicating whether to use pretrained weights or random initialization')
    parser.add_argument('--schedule', nargs="?", type=str2bool, default=True,
                        help='A flag indicating whether to use a learning rate scheduler')
    parser.add_argument('--only_output_results', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether to not run experiments and only output results.')


    args = parser.parse_args()
    if args.filepath_to_arguments_json_file is not None:
        args = extract_args_from_json(json_file_path=args.filepath_to_arguments_json_file, existing_args_dict=args)


    # arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    # print(arg_str)
    return args


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, existing_args_dict=None):
    summary_filename = json_file_path
    with open(summary_filename) as f:
        arguments_dict = json.load(fp=f)

    for key, value in vars(existing_args_dict).items():
        if key not in arguments_dict:
            arguments_dict[key] = value

    arguments_dict = AttributeAccessibleDict(arguments_dict)

    return arguments_dict
