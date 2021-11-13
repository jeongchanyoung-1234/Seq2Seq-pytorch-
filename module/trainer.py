import time
from pprint import PrettyPrinter

import torch
from torch.nn.utils import clip_grad_norm_
from torchmetrics import Accuracy


def get_norm(params):
    p_norm, g_norm = 0, 0
    for p in params:
        p_norm += torch.pow(torch.pow(p, 2).sum(), 1/2).item()
        g_norm += torch.pow(torch.pow(p.grad, 2).sum(), 1/2).item()
    return p_norm, g_norm

class Seq2SeqTrainer:
    def __init__(self,
                 config,
                 model,
                 optimizer,
                 criterion,
                 train_dataloader,
                 valid_dataloader,
                 vocab):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.vocab = vocab

        self.best_accuracy = 0
        self.best_epoch = 0

    def save_model(self, path='./seq2seq/data/model/') -> None:
        torch.save({'state_dict': self.model.state_dict(),
                    'config': self.config,
                    'src_vocab': self.vocab[0],
                    'tgt_vocab': self.vocab[1]
                    },
                   path + 'seq2seq.{}.bs{}.ed{}.hs{}.ep{}.pth'.format(
            self.config.file_fn[:-4],
            self.config.batch_size,
            self.config.embedding_dim,
            self.config.hidden_size,
            self.best_epoch
        ))

    def print_config(self) -> None:
        pp = PrettyPrinter(indent=4)
        pp.pprint(self.config)

    def train(self) -> None:
        if self.config.verbose < 1:
            self.print_config()

        total_train_loss, total_train_acc, train_cnt = 0, 0, 0
        total_valid_loss, total_valid_acc, valid_cnt = 0, 0, 0

        start = time.time()
        for epoch in range(1, self.config.epochs + 1):
            # train loop
            self.model.train()
            for batch_x, batch_y in self.train_dataloader:
                if self.config.reverse_input:
                    idx = torch.arange(batch_x.size(-1) - 1, -1, -1).long()
                    batch_x = torch.index_select(batch_x, -1, idx).contiguous()

                y_hat = self.model(batch_x, batch_y)
                loss = self.criterion(y_hat.contiguous().reshape(-1, y_hat.size(-1)), batch_y[:, 1:].contiguous().view(-1)) # (b * l, v)
                acc = Accuracy()(y_hat.contiguous().reshape(-1, y_hat.size(-1)), batch_y[:, 1:].contiguous().view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_train_loss += loss
                total_train_acc += acc
                train_cnt += 1

            avg_train_loss = total_train_loss / train_cnt
            avg_train_acc = total_train_acc / train_cnt

            # valid loop
            self.model.eval()
            for batch_x, batch_y in self.valid_dataloader:
                with torch.no_grad():
                    if self.config.reverse_input :
                        idx = torch.arange(batch_x.size(-1) - 1, -1, -1).long()
                        batch_x = torch.index_select(batch_x, -1, idx).contiguous()
                    y_hat = self.model(batch_x, batch_y)
                    loss = self.criterion(y_hat.contiguous().view(-1, y_hat.size(-1)), batch_y[:, 1 :].contiguous().view(-1)) # num_classes가 먼저 와야함!
                    acc = Accuracy()(y_hat.contiguous().view(-1, y_hat.size(-1)), batch_y[:, 1 :].contiguous().view(-1)) #Constructor의 내부에서 second argument를 boolean으로 사용해서 오류

                total_valid_loss += loss
                total_valid_acc += acc
                valid_cnt += 1

            avg_valid_loss = total_valid_loss / valid_cnt
            avg_valid_acc = total_valid_acc / valid_cnt

            if avg_valid_acc > self.best_accuracy:
                self.best_accuracy = avg_valid_acc
                self.best_epoch = epoch
                if self.config.save_model: self.save_model()

            p_norm, g_norm = get_norm(self.model.parameters())

            if epoch % self.config.verbose == 0:
                print('''|EPOCH ({}/{})| train_loss={:.4f} train_acc={:2.2f}% valid_loss={:.4f} valid_acc={:2.2f}% best_acc={:2.2f}% p_norm={:.4f} g_norm={:.4f} ({:2.2f}sec)'''.format(
                    epoch,
                    self.config.epochs,
                    avg_train_loss,
                    avg_train_acc * 100,
                    avg_valid_loss,
                    avg_valid_acc * 100,
                    self.best_accuracy * 100,
                    p_norm, g_norm,
                    time.time() - start
                ))

            total_train_loss, total_train_acc, train_cnt = 0, 0, 0
            total_valid_loss, total_valid_acc, valid_cnt = 0, 0, 0

        print('''
        |Training completed|
        Best epoch={}
        Best Accuracy={:2.2f}%
        '''.format(self.best_epoch, self.best_accuracy * 100))



