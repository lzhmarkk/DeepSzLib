import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from evaluate import evaluate
from tensorboardX import SummaryWriter
from utils.dataloader import get_dataloader
from utils.utils import Logger, Timer, EarlyStop, set_random_seed
from utils.parser import parse, get_model, get_optimizer, get_loss, get_scheduler

if __name__ == '__main__':
    args = parse()
    set_random_seed(args.seed)
    print(args)

    # save folder
    save_folder = os.path.join('./saves', args.dataset, args.model, args.exp_id)
    os.makedirs(save_folder, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_folder, 'log.txt'))
    run_folder = os.path.join(save_folder, 'run')
    os.makedirs(run_folder, exist_ok=True)
    writer = SummaryWriter(run_folder)
    timer = Timer()
    early_stop = EarlyStop(args, model_path=os.path.join(save_folder, 'best-model.pt'))

    # load data
    train_loader, val_loader, test_loader = get_dataloader(args)

    # read model
    model = get_model(args)
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
        for i, (x, y, p) in enumerate(tqdm_loader):
            model.train()
            optimizer.zero_grad()

            pred = model(x)
            if args.task == 'pred':
                los = loss(x, y)
            elif args.task == 'cls':
                los = loss(x, p)
            else:
                los = loss(x, y, p)
            los.backward()
            train_loss.append(los)
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            scheduler.step()

            tqdm_loader.set_description('Iter: {:03d}, Train Loss: {:.4f}'.format(i, train_loss[-1]))

        train_loss = np.mean(train_loss).item()
        timer.tick('train')
        print('Epoch: {:03d}, Train Time: {:.4f} secs'.format(epoch, timer.get('train')))

        # validation
        val_loss, scores, _, _ = evaluate(model, loss, val_loader)
        timer.tick('val')
        print('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(epoch, timer.get('val')))

        # post epoch
        print('Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.2f}s/epoch, Inference Time: {:.2f}s/epoch'.
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
    _, valid_scores, _, _ = evaluate(model, loss, val_loader)

    # test model
    _, test_scores, pred, tgt = evaluate(model, loss, test_loader)
    print('Test results:')
    print(json.dumps(test_scores, indent=4))
    with open(os.path.join(save_folder, 'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, indent=4)
    np.savez(os.path.join(save_folder, 'test-results.npz'), predictions=pred, targets=tgt)
