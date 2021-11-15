import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from module.model import Classifier
from module.data_loader import ClsDataLoader


def define_argparser():
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

def main(config):
    dataloader = ClsDataLoader(config)
    train_dataloader, valid_dataloader = dataloader.get_dataloader()
    src_vocab_size = len(dataloader.src.vocab)
    print(src_vocab_size)

    model = Classifier(config, src_vocab_size)

    checkpoint_callback = ModelCheckpoint(
        dirpath='C:/Users/JCY/pythonProject1/seq2seq/data/model/cls',
        filename='{epoch:02d}-{train_acc:.2f}',
    )
    trainer = pl.Trainer(
        gradient_clip_val=config.max_grad_norm,
        max_epochs=config.epochs,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloader, valid_dataloader)

if __name__ == '__main__':
    config = define_argparser()
    main(config)