import argparse

from src import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments', fromfile_prefix_chars='@', allow_abbrev=False)
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    subparsers = parser.add_subparsers(dest='mode', help='Mode of the process: train or test')

    train_parser = subparsers.add_parser('train', help='Training phase')
    eval_parser = subparsers.add_parser('eval', help='Evaluation phase')

    train_parser.add_argument('--name', default='duydz', type=str)
    train_parser.add_argument('--age', type=int, nargs='+')
    train_parser.add_argument('--bool_args', action='store_true')
    train_parser.add_argument('--metrics', type=str, nargs='+')
    train_parser.add_argument('--evaluation_info', type=str, default=['loss', 'metrics'], nargs='+',
                              choices=['loss', 'metrics'])

    args = parser.parse_args()
    print(args.name)
    print(args.age)
    print(type(args.age))
    print(args.bool_args)
    print(args.metrics)
    print(type(args.metrics))
    print(args.evaluation_info)
    print('loss' in args.evaluation_info)
