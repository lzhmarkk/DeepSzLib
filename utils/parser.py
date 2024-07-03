import os
import sys
import json
import shutil
import argparse
from utils.utils import Logger
from tensorboardX import SummaryWriter
from utils.utils import Timer, EarlyStop


def parse():
    parser = argparse.ArgumentParser()

    # pretrain
    parser.add_argument("--pretrain", action='store_true')

    # model
    parser.add_argument("--hidden", type=int, help="Hidden dimension", required=True)

    # general
    parser.add_argument("--name", type=str, help="Experiment name", required=True)
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset name", default="FDUSZ")
    parser.add_argument("--setting", type=str, choices=['Transductive', 'Inductive'], default='Transductive')
    parser.add_argument('--task', type=str, nargs='+', help='Task: (onset_)detection, prediction, classification',
                        choices=['detection', 'onset_detection', 'prediction', 'classification'], default=['detection'])
    parser.add_argument("--device", type=int, help="Device, -1 for cpu", default=-1)
    parser.add_argument("--seed", type=int, help="Random seed", default=1234)
    parser.add_argument("--runs", type=int, help="Number of runs", default=1)
    parser.add_argument("--debug", help="Debug mode", action='store_true')
    parser.add_argument("--threshold", type=float, help="Decision threshold. None for auto", default=None)
    parser.add_argument("--metric", help="Early stop metric", choices=['auc', 'f1', 'loss'], default='auc')

    # data
    parser.add_argument("--preprocess", type=str, help="raw or fft", default='fft')
    parser.add_argument("--split", type=str, help="Percentile to split train/val/test sets", default="7/1/2")
    parser.add_argument("--no_norm", help="Do NOT use z-normalizing", action='store_true')
    parser.add_argument("--window", type=int, help="Lookback window for detection (s)", default=30)
    parser.add_argument("--horizon", type=int, help="Future predict horizon (s)", default=30)
    parser.add_argument("--stride", type=int, help="Window moving stride (s)", default=30)
    parser.add_argument('--onset_history_len', help="Lookback window for onset detection (s)", type=int, default=15)
    parser.add_argument("--patch_len", type=float, help="Patch length (s)", default=1)

    # dataloader
    parser.add_argument("--n_worker", type=int, help="Number of dataloader workers", default=8)
    parser.add_argument("--pin_memory", help="Load all data into memory", action='store_true')
    parser.add_argument("--shuffle", type=bool, help="Shuffle training set", default=True)
    parser.add_argument("--argument", help="Data argument (flip and scale)", action='store_true')
    parser.add_argument("--balance", type=int, help="Balance the training set (n_neg/n_pos)", default=1)

    # loss
    parser.add_argument("--detection_loss", type=str, help="Detection loss function", default="BCE")
    parser.add_argument("--onset_detection_loss", type=str, help="Onset Detection loss function", default="BCE")
    parser.add_argument("--classification_loss", type=str, help="Classification loss function", default="CE")
    parser.add_argument("--prediction_loss", type=str, help="Prediction loss function", default="MSE")
    parser.add_argument("--lamb", type=float, help="Weight for prediction loss", default=1.0)

    # training
    parser.add_argument("--patience", type=int, help="Early stop patience", default=20)
    parser.add_argument("--epochs", type=int, help="Maximum epoch", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=256)
    parser.add_argument("--optim", type=str, help="Optimizer", default='Adam')
    parser.add_argument("--scheduler", type=str, help="Scheduler", default='Cosine')
    parser.add_argument("--grad_clip", type=float, help="Gradient clip", default=5.0)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="Weight decay", default=5e-4)

    args, unknown_args = parser.parse_known_args()
    if 'classification' in args.task:
        assert args.setting == 'Inductive', "Please use Inductive setting in classification task"
    args = parse_model_config(args, args.model)
    args = parse_unknown_config(args, unknown_args)

    args.backward = True  # default. Set false for not-training methods
    args.data_loaded = False
    assert not ('detection' in args.task and 'onset_detection' in args.task)

    args.device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    return args


def init_global_env(args):
    if 'classification' in args.task:
        save_folder = os.path.join('./saves', args.dataset + '-' + 'Classification', args.model, args.name)
        data_folder = f"./data/{args.dataset}" + '-' + 'Classification'
    else:
        save_folder = os.path.join('./saves', args.dataset + '-' + args.setting, args.model, args.name)
        data_folder = f"./data/{args.dataset}" + '-' + args.setting

    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_folder, 'log.txt'))
    args.save_folder = save_folder
    args.data_folder = data_folder
    print(args)


def init_ssl_global_env(args):
    pretrain_folder = os.path.join('./saves_ssl', args.dataset + '-' + args.setting, args.model, args.name)
    data_folder = f"./data/{args.dataset}" + '-' + args.setting

    shutil.rmtree(pretrain_folder, ignore_errors=True)
    os.makedirs(pretrain_folder, exist_ok=True)
    sys.stdout = Logger(os.path.join(pretrain_folder, 'log.txt'))
    args.pretrain_folder = pretrain_folder
    args.data_folder = data_folder
    print(args)


def init_run_env(args, run_id):
    run_folder = os.path.join(args.save_folder, f'run-{run_id}')
    os.makedirs(run_folder, exist_ok=True)
    args.writer = SummaryWriter(run_folder)
    args.timer = Timer()
    args.early_stop = EarlyStop(args, model_path=os.path.join(args.save_folder, f'best-model-{run_id}.pt'))


def parse_model_config(args, model):
    with open(f"./models/{model}/config.json", 'r') as f:
        model_cfg = json.load(f)

    data_name = args.dataset
    setting_name = args.setting
    task = args.task[0]

    match_patterns = [data_name + '-' + setting_name + '-' + task,
                      data_name + '-' + setting_name,
                      data_name,
                      setting_name,
                      'default']

    for k, v in model_cfg.items():
        if isinstance(v, dict):
            for pattern in match_patterns:
                if pattern in v:
                    v = v[pattern]
                    break
        setattr(args, k, v)

    return args


def parse_unknown_config(args, unknown_args):
    def convert_str(input_str):
        try:
            return int(input_str)
        except ValueError:
            pass

        try:
            return float(input_str)
        except ValueError:
            pass

        if input_str.lower() == 'true':
            return True
        elif input_str.lower() == 'false':
            return False

        return input_str

    unknown_args_dict = {}
    for i in range(0, len(unknown_args), 2):
        arg, value = unknown_args[i], unknown_args[i + 1]
        assert arg.startswith(("--", "-"))
        key = arg.lstrip('-')
        value = convert_str(value)
        unknown_args_dict[key] = value

    print("Unknown arguments", unknown_args_dict)

    for k, v in unknown_args_dict.items():
        setattr(args, k, v)

    return args
