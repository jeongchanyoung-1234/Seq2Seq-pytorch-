import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator


class DataLoader:
    def __init__(self,
                 config):
        self.config = config
        self.file_fn = config.file_fn
        self.src =  Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
        )
        self.tgt = Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            eos_token='<eos>',
            is_target=True,
        )

    def get_dataloader(self, root='C:/Users/JCY/pythonProject1/seq2seq/data/'):
        fields = [('src', self.src), ('tgt', self.tgt)]
        train_dataset, valid_dataset = TabularDataset(
            path=root + self.file_fn,
            format='csv',
            fields=fields
        ).split(split_ratio=0.9)

        self.src.build_vocab(train_dataset)
        self.tgt.build_vocab(train_dataset)

        train_dataloader = BucketIterator(
            train_dataset,
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.src),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            train=True,
            shuffle=True,
            sort=True,
            sort_within_batch=True
        )

        valid_dataloader = BucketIterator(
            valid_dataset,
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.src),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            train=False,
            shuffle=False,
            sort=True,
            sort_within_batch=True
        )

        return train_dataloader, valid_dataloader

if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.batch_size = 16

    config = Config()
    dataloader = DataLoader(config)
    train_dataloader, valid_dataloader = dataloader.get_dataloader()

    print(dataloader.tgt.vocab.itos[58], dataloader.tgt.vocab.itos[15], dataloader.tgt.vocab.itos[21], dataloader.tgt.vocab.itos[2])
