from typing import Optional, List

import argparse
from pathlib import Path
import torch as th

ARGS_TO_SAVE_LOAD = ['encoding_fn']

def get_arguments(additional_args: Optional[List] = None):
    """
    Arguments for training agents
    :return:
    """
    additional_args = additional_args if additional_args is not None else []

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--layout-names', help='Overcooked maps to use')
    parser.add_argument('--horizon', type=int, default=400, help='Max timesteps in a rollout')
    parser.add_argument('--num_stack', type=int, default=3, help='Number of frame stacks to use in training if frame stacks are being used')
    parser.add_argument('--encoding-fn', type=str, default='OAI_egocentric',
                        help='Encoding scheme to use. '
                             'Options: "dense_lossless", "OAI_lossless", "OAI_feats", "OAI_egocentric"')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate used in imitation learning. lr for rl is defined in rl.py')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size used in imitation learning. bs for rl is defined in rl.py')

    parser.add_argument('--ADV-seed', type=int, default=68, help='seed used in adverary-play')
    parser.add_argument('--ADV-h-dim', type=int, default=512, help='hidden dimension used in adverary-play')

    parser.add_argument('--algo-name', type=str)
    parser.add_argument('--num-players', type=int, help='Number of players in the game')
    parser.add_argument('--pop-force-training', type=str2bool, help='Flag indicating whether or not to force training of the population')
    parser.add_argument('--primary-force-training', type=str2bool, help='Flag indicating whether or not to force training of the primary agent')
    parser.add_argument('--adversary-force-training', type=str2bool, help='Flag indicating whether or not to force training of the adversary agent')
    parser.add_argument('--how-long', type=int)

    parser.add_argument('--exp-name', type=str, default='last',
                        help='Name of experiment. Used to tag save files.')
    parser.add_argument('--base-dir', type=str, default=Path.cwd(),
                        help='Base directory to save all models, data, wandbai.')
    parser.add_argument('--data-path', type=str, default='data',
                        help='Path from base_dir to where the human data is stored')
    parser.add_argument('--dataset', type=str, default='2019_hh_trials_all.pickle',
                        help='Which set of human data to use. '
                             'See https://github.com/HumanCompatibleAI/human_aware_rl/tree/master/human_aware_rl/static/human_data for options')

    parser.add_argument('--wandb-mode', type=str, default='online',
                        help='Wandb mode. One of ["online", "offline", "disabled"')
    parser.add_argument('--wandb-ent', type=str,
                        help='Wandb entity to log to.')
    parser.add_argument('--sb-verbose', type=int, default=1)
    parser.add_argument('-c', type=str, default='', help='for stupid reasons, but dont delete')
    parser.add_argument('args', nargs='?', type=str, default='', help='')

    parser.add_argument('--epoch-timesteps', type=int)
    parser.add_argument('--n-envs', type=int, help='Number of environments to use while training')
    parser.add_argument('--teammates-len',  type=int)
    parser.add_argument('--overcooked-verbose', type=str2bool, default=False, help="Disables the overcooked game logs")

    parser.add_argument('--pop-total-training-timesteps', type=int)
    parser.add_argument('--fcp-total-training-timesteps', type=int)
    parser.add_argument('--fcp-w-sp-total-training-timesteps', type=int)
    parser.add_argument('--eval-steps-interval', type=int, default=40)
    parser.add_argument('--reward-magnifier', type=float, default=3.0)

    parser.add_argument('--parallel', type=str2bool, default=True)

    parser.add_argument('--dynamic-reward', type=str2bool, default=True)
    parser.add_argument('--final-sparse-r-ratio', type=float, default=0.5)
    parser.add_argument('--prioritized-sampling', type=str2bool, default=False, help='Flag indicating whether or not to use prioritized sampling for teammate selection, if True the worst performing teammates will be prioritized')
    parser.add_argument('--exp-dir', type=str, help='Folder to save/load experiment result')

    parser.add_argument('--primary-learner-type', type=str, default='originaler')
    parser.add_argument('--adversary-learner-type', type=str, default='selfisher')
    parser.add_argument('--pop-learner-type', type=str, default='originaler')
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)
    parser.add_argument("--exp-name-prefix", default="", type=str)

    parser.add_argument("--num-of-ckpoints", type=int, default=15)
    parser.add_argument("--save-freq", type=int, default=100000, help='Frequency for saving checkpoints')
    parser.add_argument("--resume", action="store_true", default=False, help="Restart from last checkpoint for population training only")

    parser.add_argument("--use-val-func-for-heatmap-gen", type=str2bool, default=False)
    parser.add_argument("--num-eval-for-heatmap-gen", type=int, default=2)
    parser.add_argument("--num-static-advs-per-heatmap", type=int, default=1)
    parser.add_argument("--num-dynamic-advs-per-heatmap", type=int, default=1)
    parser.add_argument("--num-steps-in-traj-for-dyn-adv", type=int, default=2)
    parser.add_argument("--custom-agent-ck-rate-generation", type=int)

    parser.add_argument('--gen-pop-for-eval', type=str2bool, default=False, help="Specifies whether to generate a population of agents for evaluation purposes. Currently, this functionality is limited to self-play agents, as support for other methods has not yet been implemented..)")
    parser.add_argument("--total-ego-agents", type=int, default=4)
    parser.add_argument("--ck-list-offset", type=int, default=0)


    for parser_arg, parser_kwargs in additional_args:
        parser.add_argument(parser_arg, **parser_kwargs)

    # Use parse_known_args to avoid errors when external callers (e.g., Sacred) pass
    # additional CLI flags that this parser doesn't recognize.
    args, _unknown = parser.parse_known_args()
    args.base_dir = Path(args.base_dir)
    args.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    if isinstance(args.layout_names, str):
        args.layout_names = args.layout_names.split(',')


    return args

def get_args_to_save(curr_args):
    arg_dict = vars(curr_args)
    arg_dict = {k: v for k, v in arg_dict.items() if k in ARGS_TO_SAVE_LOAD}
    return arg_dict

def set_args_from_load(loaded_args, curr_args):
    for arg in ARGS_TO_SAVE_LOAD:
        setattr(curr_args, arg, loaded_args[arg])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
