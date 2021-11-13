import argparse

import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator

from model import Seq2Seq
from trainer import Seq2SeqTrainer
from data_loader import DataLoader


optimizer_map = {'sgd': optim.SGD, 'adam': optim.Adam, 'rmsprop': optim.RMSprop}


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--batch_size', type=int, default=128,
                   help='the size of mini-batches')
    p.add_argument('--hidden_size', type=int, default=128,
                   help='the size of hidden states')
    p.add_argument('--embedding_dim', type=int, default=16,
                   help='the output size of embedding layer')
    p.add_argument('--n_layers', type=int, default=2,
                   help='the number of layers in both encoder and decoder')
    p.add_argument('--dropout', type=float, default=.5,
                   help='the ratio of dropout')
    p.add_argument('--optimizer', type=str, default='adam',
                   help='Adam/SGD/RMSprop supported')
    p.add_argument('--lr', type=float, default=1e-2,
                   help='the learning rate')
    p.add_argument('--max_grad_norm', type=float, default=5.,
                   help='gradient clipping')
    p.add_argument('--epochs', type=int, default=5,
                   help='the number of training the whole batches')
    p.add_argument('--verbose', type=int, default=1,
                   help='print traning logs every n-epochs')
    p.add_argument('--reverse_input', action='store_true',
                   help='reversing the input, which is suggested at GNMT')
    p.add_argument('--file_fn', type=str, default='date.csv',
                   help='training source filename')

    config = p.parse_args()
    return config

def main(config):
    dataloader = DataLoader(config)
    train_dataloader, valid_dataloader = dataloader.get_dataloader()
    vocab = dataloader.src.vocab, dataloader.tgt.vocab

    src_vocab_size = len(vocab[0])
    tgt_vocab_size = len(vocab[1])
    print(dataloader.tgt.vocab.itos)

    print('|train|={} |valid|={}'.format(
        len(train_dataloader) * config.batch_size,
        len(valid_dataloader) * config.batch_size)
    )
    print('|Source vocabulary|:', src_vocab_size)
    print('|Target vocabulary|:', tgt_vocab_size)

    model = Seq2Seq(config, src_vocab_size, tgt_vocab_size)
    optimizer = optimizer_map[config.optimizer.lower()](model.parameters(), lr=config.lr)
    criterion = nn.NLLLoss()

    trainer = Seq2SeqTrainer(
        config,
        model,
        optimizer,
        criterion,
        train_dataloader,
        valid_dataloader,
        vocab
    )

    trainer.train()

if __name__ == '__main__':
    config = define_argparser()
    main(config)




