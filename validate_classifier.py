import argparse

import torch

from module.model import Classifier
from module.data_loader import ClsDataLoader


def define_train_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--max_grad_norm', type=float, default=5.)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--embedding_dim', type=int, default=16)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--dropout', type=float, default=.2)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--file_fn', type=str, default='classify.csv')
    config = p.parse_args()

    return config

config = define_train_argparser()
dataloader = ClsDataLoader(config)
train_dataloader, valid_dataloader = dataloader.get_dataloader()

src_vocab_size = len(dataloader.src.vocab)

model = Classifier.load_from_checkpoint(
    'C:/Users/JCY/pythonProject1/seq2seq/data/model/cls/epoch=01-train_acc=1.00.ckpt'
)


model.eval()
input = 'a p r i l / 0 7 / 1'
input = '1 1 1 + 7 5 <pad> <pad> <pad> <pad>'
result = []
for c in input.split():
    result.append(dataloader.src.vocab.stoi[c])
print(result)
y_hat = model(torch.Tensor(result).long().unsqueeze(0))
print(y_hat.argmax(-1).item())