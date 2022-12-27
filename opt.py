import argparse
from pathlib import Path
def get_opts():
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument(
        '--mode',
        type=str,
        default='normal',
    )

    # dataset hparam
    parser.add_argument(
        '--time_step',
        type=int,
        default=6,
    )

    # model hparams
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=512,
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=3
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
    )
    # trainging hparams
    parser.add_argument(
        '--lr',
        type=int,
        default=1e-4,
    )
    parser.add_argument(
        '--exp_name', 
        type=str,
        default='exp',
    )                   
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=500,
    )
    parser.add_argument(
        '--optim',
        type=str,
        default='Adam'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='StepLR',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
    )

    # dir path
    parser.add_argument(
        '--ckpt_dir',
        type=Path,
        default='./ckpts'
    )
    parser.add_argument(
        '--log_dir',
        type=Path,
        default='./logs'
    )

    args = parser.parse_args()

    return args
