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
from utils.loss import get_loss


def main(args, run_id):
    run_folder = os.path.join(args.save_folder, 'run')
    os.makedirs(run_folder, exist_ok=True)
    writer = SummaryWriter(run_folder)
    timer = Timer()
    early_stop = EarlyStop(args, model_path=os.path.join(args.save_folder, f'best-model-{run_id}.pt'))

    # load data
    train_loader, val_loader, test_loader = get_dataloader(args)

    # read model
    model = get_model(args).to(args.device)
    print(model)
    print('Number of model parameters is', sum([p.nelement() for p in model.parameters()]))
    loss = get_loss(args)
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
        early_stop.step(epoch, val_loss, model)

    print("Average Training Time: {:.4f} secs/epoch".format(timer.get_all('train')))
    print("Average Inference Time: {:.4f} secs".format(timer.get_all('val')))

    # validate model
    model = early_stop.load_best_model()
    _, valid_scores, _, _ = evaluate(args, 'val', model, loss, val_loader)

    # test model
    _, test_scores, pred, tgt = evaluate(args, 'test', model, loss, test_loader)
    print(f'Test results of run {run_id}:')
    print(json.dumps(test_scores, indent=4))
    with open(os.path.join(args.save_folder, f'test-scores-{run_id}.json'), 'w+') as f:
        json.dump(test_scores, f, indent=4)
    np.savez(os.path.join(args.save_folder, f'test-results-{run_id}.npz'), predictions=pred, targets=tgt)

    return test_scores


if __name__ == '__main__':
    args = parse()

    # save folder
    save_folder = os.path.join('./saves', args.dataset, args.model, args.name)
    os.makedirs(save_folder, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_folder, 'log.txt'))
    args.save_folder = save_folder
    print(args)

    # run for several runs
    test_scores_multiple_runs = []
    for run in range(args.runs):
        set_random_seed(args.seed + run)
        test_scores = main(args, run)
        test_scores_multiple_runs.append(test_scores)

    # merge results from several runs
    test_scores = {}
    for k in test_scores_multiple_runs[0].keys():
        test_scores[k] = np.mean([scores[k] for scores in test_scores_multiple_runs]).item()
    print(f"Average test results of {args.runs} runs:")
    print(json.dumps(test_scores, indent=4))
    with open(os.path.join(args.save_folder, 'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, indent=4)

    print(f"name: {args.name}")
    for k in test_scores:
        print(f"{k}\t", end='')
    print()
    for k in test_scores:
        print("{:.4f}\t".format(test_scores[k]), end='')
    print()
