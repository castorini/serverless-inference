import time
import os
import numpy as np
import random

import sys
import torch
import torch.nn as nn
import torch.onnx

from sm_cnn.args import get_args
from sm_cnn.model import SmPlusPlus
from utils.relevancy_metrics import get_map_mrr
from datasets.trecqa import TRECQA
from datasets.wikiqa import WikiQA

args = get_args()
config = args

torch.manual_seed(args.seed)

# Set default configuration in : args.py
args = get_args()
config = args

# Set random seed for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("You have Cuda but you're using CPU for training.")
np.random.seed(args.seed)
random.seed(args.seed)

word_vectors_dir = os.path.join(os.pardir, os.pardir, 'data', 'word2vec')
word_vectors_name = 'aquaint+wiki.txt.gz.ndim=50.txt'

if config.dataset not in ('TREC', 'wiki'):
    print("Unsupported dataset")
    sys.exit(1)

if config.dataset == 'TREC':
    dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'TrecQA/')
    train_iter, dev_iter, test_iter = TRECQA.iters(dataset_root, word_vectors_name, word_vectors_dir, args.batch_size, device=args.gpu)
    embedding_dim = TRECQA.TEXT_FIELD.vocab.vectors.size()
    embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
    embedding.weight = nn.Parameter(TRECQA.TEXT_FIELD.vocab.vectors)
else:
    dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'WikiQA/')
    train_iter, dev_iter, test_iter = WikiQA.iters(dataset_root, word_vectors_name, word_vectors_dir, args.batch_size, device=args.gpu)
    embedding_dim = WikiQA.TEXT_FIELD.vocab.vectors.size()
    embedding = nn.Embedding(embedding_dim[0], embedding_dim[1])
    embedding.weight = nn.Parameter(WikiQA.TEXT_FIELD.vocab.vectors)

embedding.weight.requires_grad = False

if args.gpu != -1:
    with torch.cuda.device(args.gpu):
        embedding = embedding.cuda()

print("Dataset {}".format(args.dataset))
print("Train instance", len(train_iter.dataset.examples))
print("Dev instance", len(dev_iter.dataset.examples))
print("Test instance", len(test_iter.dataset.examples))

if args.resume_snapshot:
    if args.cuda:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = SmPlusPlus(config)

    if args.cuda:
        model.cuda()
        print("Shift model to GPU")


parameter = filter(lambda p: p.requires_grad, model.parameters())

# the SM model originally follows SGD but Adadelta is used here
optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
early_stop = False
best_dev_map = 0
iterations = 0
iters_not_improved = 0
epoch = 0
start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, args.dataset), exist_ok=True)
print(header)

while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, best_dev_map))
        break
    epoch += 1
    train_iter.init_epoch()
    n_correct, n_total = 0, 0

    for batch_idx, batch in enumerate(train_iter):
        iterations += 1
        model.train()
        optimizer.zero_grad()
        question = embedding(batch.question)
        answer = embedding(batch.answer)
        scores = model(question, answer, batch.ext_feats)
        n_correct += (torch.max(scores, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total

        loss = criterion(scores, batch.label)
        loss.backward()
        optimizer.step()

        # Evaluate performance on validation set
        if iterations % args.dev_every == 1:
            # switch model into evaluation mode
            model.eval()
            dev_iter.init_epoch()
            dev_losses = []

            qids = []
            predictions = []
            labels = []
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                question = embedding(dev_batch.question)
                answer = embedding(dev_batch.answer)
                scores = model(question, answer, dev_batch.ext_feats)
                dev_loss = criterion(scores, dev_batch.label)
                dev_losses.append(dev_loss.data[0])

                qids.extend(dev_batch.id.data.cpu().numpy())
                predictions.extend(scores.data.exp()[:, 1].cpu().numpy())
                labels.extend(dev_batch.label.data.cpu().numpy())

            dev_map, dev_mrr = get_map_mrr(qids, predictions, labels)
            print(dev_log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data[0],
                                          sum(dev_losses) / len(dev_losses), train_acc, dev_map))

            # Update validation results
            if dev_map > best_dev_map:
                iters_not_improved = 0
                best_dev_map = dev_map
                snapshot_path = os.path.join(args.save_path, args.dataset, 'static_best_model.pt')
                torch.save(model, snapshot_path)
            else:
                iters_not_improved += 1
                if iters_not_improved >= args.patience:
                    early_stop = True
                    break

        if iterations % args.log_every == 1:
            # print progress message
            print(log_template.format(time.time() - start,
                                      epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
                                      train_acc, ' ' * 12))
