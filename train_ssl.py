import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from evaluate import evaluate
from tensorboardX import SummaryWriter
from utils.dataloader import get_dataloader
from utils.utils import Logger, Timer, EarlyStop, set_random_seed, to_gpu
from utils.parser import parse, get_model, get_optimizer, get_scheduler
from utils.loss import MyLoss


def main(args, run_id=0, fine_tune_stage=False):
    print("#" * 30)
    print("#" * 12 + f"   {run_id}   " + "#" * 12)
    print("#" * 30)
    train_loader, val_loader, test_loader = get_dataloader(args)
    timer = Timer()
    early_stop = EarlyStop(args, model_path=os.path.join(args.pretrain_folder, f'best-model.pt'))

    if fine_tune_stage:
        run_folder = os.path.join(args.save_folder, 'run')
        os.makedirs(run_folder, exist_ok=True)
        writer = SummaryWriter(run_folder)

        model = early_stop.load_best_model()
        model.task = args.task
        early_stop = EarlyStop(args, model_path=os.path.join(args.save_folder, f'best-model-{run_id}.pt'))
    else:
        run_folder = os.path.join(args.pretrain_folder, 'run')
        os.makedirs(run_folder, exist_ok=True)
        writer = SummaryWriter(run_folder)

        model = get_model(args).to(args.device)
    print(model)
    print('Number of model parameters is', sum([p.nelement() for p in model.parameters()]))
    loss = MyLoss(args)
    optimizer = get_optimizer(args, model.parameters())
    scheduler = get_scheduler(args, optimizer)

    # train model
    print("Start training...", flush=True)
    timer.flush()
    for epoch in range(args.epochs):
        # train
        train_loss = []
        tqdm_loader = tqdm(train_loader, ncols=150)
        for i, (u, x, y, p) in enumerate(tqdm_loader):
            model.train()

            if args.backward:
                optimizer.zero_grad()

            x, y, p = to_gpu(x, y, p, device=args.device)
            z = model(x, p, y)
            los = loss(z, p, y)

            if args.backward:
                los.backward()

                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()

            train_loss.append(los.item())
            tqdm_loader.set_description('Iter: {:03d}, Train Loss: {:.4f}, lr: {:.7f}'
                                        .format(i, train_loss[-1], scheduler.get_last_lr()[0]))

        train_loss = np.mean(train_loss).item()
        timer.tick('train')
        print('Epoch: {:03d}, Train Time: {:.4f} secs'.format(epoch, timer.get('train')))

        # validation
        val_loss, scores, _, _ = evaluate(args, 'train', model, loss, val_loader)
        timer.tick('val')
        print('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(epoch, timer.get('val')))

        # post epoch
        print('Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, '
              'Training Time: {:.2f}s/epoch, Inference Time: {:.2f}s/epoch'.
              format(epoch, train_loss, val_loss, timer.get('train'), timer.get('val')), flush=True)
        writer.add_scalars('loss', {'train': train_loss}, global_step=epoch)
        writer.add_scalars('loss', {'valid': val_loss}, global_step=epoch)

        if early_stop.now_stop(epoch):
            break
        early_stop.step(epoch, val_loss, scores, model)

    print("Average Training Time: {:.4f} secs/epoch".format(timer.get_all('train')))
    print("Average Inference Time: {:.4f} secs".format(timer.get_all('val')))

    if fine_tune_stage:
        model = early_stop.load_best_model()
        _, valid_scores, _, _ = evaluate(args, 'val', model, loss, val_loader)
        _, test_scores, pred, tgt = evaluate(args, 'test', model, loss, test_loader)
        print(f'Test results of run {run_id}:')
        print(json.dumps(test_scores, indent=4))
        with open(os.path.join(args.save_folder, f'test-scores-{run_id}.json'), 'w+') as f:
            json.dump(test_scores, f, indent=4)
        np.savez(os.path.join(args.save_folder, f'test-results-{run_id}.npz'), predictions=pred, targets=tgt, thres=args.threshold_value)

    else:
        args.task = ['cls']
        loss = MyLoss(args)
        model = early_stop.load_best_model()
        _, valid_scores, _, _ = evaluate(args, 'val', model, loss, val_loader)
        _, test_scores, pred, tgt = evaluate(args, 'test', model, loss, test_loader)
        print(f'Test results of pretraining:')
        print(json.dumps(test_scores, indent=4))
        with open(os.path.join(args.pretrain_folder, f'test-scores.json'), 'w+') as f:
            json.dump(test_scores, f, indent=4)
        np.savez(os.path.join(args.pretrain_folder, f'test-results.npz'), predictions=pred, targets=tgt, thres=args.threshold_value)

    return test_scores


if __name__ == '__main__':
    args = parse()

    # save folder
    pretrain_folder = os.path.join('./saves_ssl', args.dataset + '-' + args.setting, args.model, args.name)
    os.makedirs(pretrain_folder, exist_ok=True)
    args.pretrain_folder = pretrain_folder

    if args.pretrain:  # pre-training
        sys.stdout = Logger(os.path.join(pretrain_folder, 'log.txt'))

        assert args.pretrain
        assert args.task == ['pred']
        args.runs = 1
        args.metric = 'loss'
        args.threshold = False
        args.lamb = 1.0
        print(args)
        print("Start pretraining...")

        set_random_seed(args.seed)
        main(args)
        print(f"Pretrain ends. Save pretrained model to {os.path.join(args.pretrain_folder, f'best-model.pt')}")

    else:  # fine-tuning
        save_folder = os.path.join('./saves', args.dataset + '-' + args.setting, args.model, args.name)
        os.makedirs(save_folder, exist_ok=True)
        sys.stdout = Logger(os.path.join(save_folder, 'log.txt'))
        args.save_folder = save_folder
        print(args)

        assert args.task == ['cls']

        print("Start fine-tuning")
        test_scores_multiple_runs = []
        for run in range(args.runs):
            set_random_seed(args.seed + run)
            test_scores = main(args, run, fine_tune_stage=True)
            test_scores_multiple_runs.append(test_scores)

        # merge results from several runs
        test_scores = {'mean': {}, 'std': {}}
        for k in test_scores_multiple_runs[0].keys():
            test_scores['mean'][k] = np.mean([scores[k] for scores in test_scores_multiple_runs]).item()
            test_scores['std'][k] = np.std([scores[k] for scores in test_scores_multiple_runs]).item()

        print(f"Average test results of {args.runs} runs:")
        print(json.dumps(test_scores, indent=4))
        with open(os.path.join(args.save_folder, 'test-scores.json'), 'w+') as f:
            json.dump(test_scores, f, indent=4)

        print(f"Dataset: {args.dataset}, model: {args.model}, name: {args.name}")
        print('*' * 30, 'mean', '*' * 30)
        for k in test_scores['mean']:
            print(f"{k}\t", end='')
        print()
        for k in test_scores['mean']:
            print("{:.4f}\t".format(test_scores['mean'][k]), end='')
        print()

        print('*' * 30, 'std', '*' * 30)
        for k in test_scores['std']:
            print(f"{k}\t", end='')
        print()
        for k in test_scores['std']:
            print("{:.4f}\t".format(test_scores['std'][k]), end='')
        print()
