import numpy as np
import random
import logging
import os

import torch
import torch.nn as nn

from sm_cnn.args import get_args
from utils.relevancy_metrics import get_map_mrr
from datasets.trecqa import TRECQA
from datasets.wikiqa import WikiQA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

args = get_args()
config = args

torch.manual_seed(args.seed)

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    logger.info("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    logger.info("Warning: You have Cuda but do not use it. You are using CPU for training")
np.random.seed(args.seed)
random.seed(args.seed)

word_vectors_dir = os.path.join(os.pardir, os.pardir, 'data', 'GloVe')
word_vectors_name = 'glove.840B.300d.txt'

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

if args.cuda:
    model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = torch.load(args.trained_model, map_location=lambda storage,location: storage)


def predict(dataset, test_mode, dataset_iter):
    model.eval()
    dataset_iter.init_epoch()

    qids = []
    predictions = []
    labels = []
    for dev_batch_idx, dev_batch in enumerate(dataset_iter):
        question = embedding(dev_batch.question)
        answer = embedding(dev_batch.answer)
        scores = model(question, answer, dev_batch.ext_feats)

        qids.extend(dev_batch.id.data.cpu().numpy())
        predictions.extend(scores.data.exp()[:, 1].cpu().numpy())
        labels.extend(dev_batch.label.data.cpu().numpy())

    dev_map, dev_mrr = get_map_mrr(qids, predictions, labels)
    logger.info("{} {}".format(dev_map, dev_mrr))

# Run the model on the dev set
predict(config.dataset, 'dev', dataset_iter=dev_iter)

# Run the model on the test set
predict(config.dataset, 'test', dataset_iter=test_iter)

if args.onnx:
    print("Saving model to ONNX...")
    dummy_batch = next(iter(dev_iter))
    dummy_input = (dummy_batch.question, dummy_batch.answer, dummy_batch.ext_feats)
    torch.onnx.export(model, dummy_input, "sm_model.proto", verbose=True)
