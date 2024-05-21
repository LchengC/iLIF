# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ECI')

    # dataset
    parser.add_argument("--dataset_type", default='EventStoryLine', type=str, help='dataset name')
    parser.add_argument("--cache_path", default='./data/database', type=str)
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--inter_or_intra", default='intra_and_inter', type=str)
    parser.add_argument("--k_fold", default=5, type=int)
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--use_cache', default=True, action="store_true")
    parser.add_argument('--in_channels', default={'event': 768}, type=dict)
    parser.add_argument('--metadata', default=(['event'], [('event','intra','event'),('event','inter','event')]), type=tuple)
    parser.add_argument('--beta_inter', default=0.3, type=float)
    parser.add_argument('--beta_intra', default=0.7, type=float)

    # # model arguments
    parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='Log model name')
    parser.add_argument('--mlp_size', default=2304, type=int, help='mlp layer_size')
    parser.add_argument('--mlp_drop', default=0.4, type=float, help='mlp dropout layer')
    parser.add_argument('--n_last', default=768, type=int, help='CGE out layer')
    parser.add_argument('--threshold', default=2, type=int, help='structural difference threshold δ')
    parser.add_argument('--w', default=0.6, type=float, help='The relation confidence threshold ω')

    # # loss arguments
    parser.add_argument('--gamma', default=2.0, type=float, help='loss gamma argument')
    parser.add_argument('--no_of_classes', default=3, type=int, help='loss classes argument')
    parser.add_argument('--loss_type', default="focal", type=str, help='loss type')
    parser.add_argument('--max_iteration', default=9, type=int, help='Maximum iteration number L')
    parser.add_argument('--min_iteration', default=2, type=int, help='Minimum iteration number')
    parser.add_argument('--num_heads', default=4, type=int, help='The number of attention head K')

    # # training arguments
    parser.add_argument('--seed', default=209, type=int, help='seed for reproducibility')
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--epoch', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--t_lr', default=2e-5, type=float, help='initial transformer learning rate')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial other module learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument('--rate', default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warm_up_ratio", default=0.1, type=float)
    parser.add_argument("--class_weight", default=0.75, type=float)
    parser.add_argument('--logging_steps', default=50, type=int)

    parser.add_argument('--valid_epoch', default=1, type=int)
    parser.add_argument('--test_epoch', default=1, type=int)

    parser.add_argument('--log', default='', type=str, help='Log result file name')

    args = parser.parse_args()
    return args
