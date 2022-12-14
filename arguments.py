import argparse


def add_train_arguments(parser: argparse.ArgumentParser):
    _add_common_arguments(parser)
    _add_train_data_args(parser)
    _add_model_args(parser)
    _add_train_args(parser)


def add_eval_arguments(parser: argparse.ArgumentParser):
    _add_common_arguments(parser)
    parser.add_argument('--saved_model_path', type=str, help='Path to the trained model')
    parser.add_argument('--data_name', type=str, default='CafeFNewsRecommend', help='Name of the eval dataset')
    parser.add_argument('--eval_behaviors_path', default='data/valid_behaviors.tsv', type=str)
    parser.add_argument('--fast_eval', action='store_true')
    parser.add_argument('--eval_batch_size', type=int, help='How many samples per batch to load in the test phase')
    parser.add_argument('--dataloader_num_workers', type=int, help='How many subprocesses to use for data loading')
    parser.add_argument('--eval_path', type=str, default='eval',
                        help='Path to the directory where evaluation information is stored')


def _add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--pretrained_tokenizer', type=str)
    parser.add_argument('--user2id_path', type=str)
    parser.add_argument('--category2id_path', type=str)
    parser.add_argument('--max_title_length', type=int)
    parser.add_argument('--max_sapo_length', type=int)
    parser.add_argument('--his_length', type=int, help='Max number of user click history')
    parser.add_argument('--seed', type=int, help='Seed value')
    parser.add_argument('--save_eval_result', action='store_true')
    parser.add_argument('--metrics', type=str, nargs='+')


def _add_train_data_args(parser: argparse.ArgumentParser):
    parser.add_argument('--data_name', type=str, default=None, help='Name of the train dataset')
    parser.add_argument('--train_behaviors_path', type=str)
    parser.add_argument('--train_news_path', type=str)
    parser.add_argument('--eval_behaviors_path', type=str)
    parser.add_argument('--eval_news_path', type=str)
    parser.add_argument('--fast_eval', action='store_true')


def _add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument('--pretrained_embedding', type=str)
    parser.add_argument('--word_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--category_embed_dim', type=int)
    parser.add_argument('--num_click_embed_dim', type=int)
    parser.add_argument('--query_dim', type=int)
    parser.add_argument('--score_weight', type=float)
    parser.add_argument('--attn_dropout', type=float)
    parser.add_argument('--dropout', type=float, help='Dropout value')


def _add_train_args(parser: argparse.ArgumentParser):
    parser.add_argument('--train_path', type=str, default='train',
                        help='Path to the directory where training information is stored')
    parser.add_argument('--tensorboard_path', type=str, default='runs',
                        help='Path of the directory where to save the log files to be parsed by TensorBoard')
    parser.add_argument('--npratio', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, help='How many samples per batch to load in the training phase')
    parser.add_argument('--eval_batch_size', type=int,
                        help='How many samples per batch to load in the validation phase')
    parser.add_argument('--dataloader_drop_last', action='store_true',
                        help='Whether to drop the last incomplete batch (if the length of the dataset is not divisible '
                             'by the batch size) or not')
    parser.add_argument('--dataloader_num_workers', type=int, help='How many subprocesses to use for data loading')
    parser.add_argument('--dataloader_pin_memory', action='store_true', help='If True, the data loader will copy '
                                                                             'Tensors into device/CUDA pinned memory '
                                                                             'before returning them.')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='If set to a positive number, the total number of training steps to perform '
                             '(Overrides num_train_epochs)')
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training')
    parser.add_argument('--gradient_accumulation_steps', type=int,
                        help='Number of updates steps to accumulate the gradients for, before performing a '
                             'backward/update pass')
    parser.add_argument('--num_train_epochs', type=int, help='Total number of training epochs to perform')
    parser.add_argument('--learning_rate', type=float, help='Learning rate of the optimization algorithm')
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help='Ratio of total training steps used for a linear warmup from 0 to learning_rate')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Number of steps used for a linear warmup from 0 to learning_rate (override warmup_ratio)')
    parser.add_argument('--max_grad_norm', type=float, help='Max norms of the gradients')
    parser.add_argument('--weight_decay', type=float, help='Decoupled weight decay to apply')
    parser.add_argument('--logging_steps', type=int, help='Number of update steps between two logs')
    parser.add_argument('--evaluation_info', type=str, nargs='+', choices=['loss', 'metrics'])
    parser.add_argument('--eval_steps', type=int, help='Number of update steps between two evaluations')
