import json
import argparse
import os.path


def parse():
    parser = argparse.ArgumentParser()

    # pretrain
    parser.add_argument("--stage", choices=['train', 'pretrain', 'finetune'], default='train')

    # model
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--hidden", type=int, help="Hidden dimension", required=True)

    # general
    parser.add_argument("--save_folder", type=str, help="Model save path", required=True)
    parser.add_argument("--data_folder", type=str, help="Dataset path", required=True)
    parser.add_argument('--task', type=str, nargs='+', help='Task: (onset_)detection, prediction, classification',
                        choices=['detection', 'onset_detection', 'prediction', 'classification'], default=['detection'])
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--runs", type=int, help="Number of runs", default=1)
    parser.add_argument("--debug", help="Debug mode", action='store_true')
    parser.add_argument("--threshold", type=float, help="Decision threshold. None for auto", default=None)
    parser.add_argument("--metric", help="Early stop metric", choices=['auc', 'f1', 'loss'], default='auc')

    # data
    parser.add_argument("--preprocess", type=str, choices=['raw', 'fft'], default='fft')
    parser.add_argument("--split", type=str, help="Percentile to split train/val/test sets", default="7/1/2")
    parser.add_argument("--norm", type=str, help="Use z-normalizing", default='true')
    parser.add_argument("--window", type=int, help="Lookback window for detection (s)", default=30)
    parser.add_argument("--horizon", type=int, help="Future predict horizon (s)", default=30)
    parser.add_argument("--stride", type=int, help="Window moving stride (s)", default=30)
    parser.add_argument('--onset_history_len', help="Lookback window for onset detection (s)", type=int, default=15)
    parser.add_argument("--patch_len", type=float, help="Patch length (s)", default=1)

    # dataloader
    parser.add_argument("--n_worker", type=int, help="Number of dataloader workers", default=8)
    parser.add_argument("--pin_memory", help="Load all data into memory", action='store_true')
    parser.add_argument("--shuffle", type=str, help="Shuffle training set", default='true')
    parser.add_argument("--argument", help="Data argument (flip and scale)", action='store_true')
    parser.add_argument("--balance", type=int, help="Balance the training set (n_neg/n_pos)", default=1)

    # loss
    parser.add_argument("--detection_loss", type=str, help="Detection loss function", default="BCE")
    parser.add_argument("--onset_detection_loss", type=str, help="Onset Detection loss function", default="BCE")
    parser.add_argument("--classification_loss", type=str, help="Classification loss function", default="CE")
    parser.add_argument("--prediction_loss", type=str, help="Prediction loss function", default="MSE")
    parser.add_argument("--lamb", type=float, help="Weight for prediction loss", default=1.0)

    # training
    parser.add_argument("--seed", type=int, help="Random seed", default=1234)
    parser.add_argument("--patience", type=int, help="Early stop patience", default=20)
    parser.add_argument("--epochs", type=int, help="Maximum epoch", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=256)
    parser.add_argument("--optim", type=str, help="Optimizer", default='Adam')
    parser.add_argument("--scheduler", type=str, help="Scheduler", default='Cosine')
    parser.add_argument("--grad_clip", type=float, help="Gradient clip", default=5.0)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="Weight decay", default=5e-4)

    return parser


def parse_args(parser):
    args, unknown_args = parser.parse_known_args()

    dataset = os.path.basename(args.data_folder.rstrip('/'))
    args.dataset, args.setting = dataset.split('-')
    if 'classification' in args.task:
        args.setting = 'Inductive'
    assert args.dataset in ['FDUSZ', 'TUSZ', 'CHBMIT']
    assert args.setting in ['Transductive', 'Inductive']

    args = parse_model_config(args, args.model)
    args = parse_unknown_config(args, unknown_args)

    args.backward = True  # default. Set false for not-training methods
    args.data_loaded = False
    assert not ('detection' in args.task and 'onset_detection' in args.task)

    args.norm = args.norm.lower() == 'true'
    args.shuffle = args.shuffle.lower() == 'true'

    print(args)
    return args


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
