import math

import torch
import torch.nn as nn

import pytorch_lightning as pl
from torchmetrics import Accuracy


class Attention(nn.Module):
    def __init__(self,
                 config):
        super(Attention, self).__init__()
        self.linear = nn.Linear(in_features=config.hidden_size,
                                out_features=config.hidden_size,
                                device='cuda' if torch.cuda.is_available() else 'cpu')
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_t_dec, x_enc, mask=None):
        # |score| = (bs, 1, m)
        score = torch.bmm(
                    self.linear(h_t_dec),
                    x_enc.permute(0, 2, 1).contiguous() # view, permute, transpose
                )
        if mask is not None:
            score.masked_fill_(mask.unsqueeze(1), float('-inf'))

        w = self.softmax(score)
        context_vector = torch.bmm(w, x_enc)
        return context_vector

class Encoder(nn.Module):
    def __init__(self,
                 config,
                 vocab_size,
                 ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size // 2,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.,
            bidirectional=True,
        )

    def forward(self, x):
        x = self.embedding(x)
        x, (h, c) = self.lstm(x)
        h_tmp, c_tmp = [], []
        for i in range(0, h.size(0), 2):
            h_tmp.append(torch.cat(
                [h[i, :, :].unsqueeze(0),
                 h[i + 1, :, :].unsqueeze(0)],
                dim=-1)) # cat과 stack의 차이점?
            c_tmp.append(torch.cat(
                [h[i, :, :].unsqueeze(0),
                 h[i + 1, :, :].unsqueeze(0)],
                dim=-1))
        h_new, c_new = torch.cat(h_tmp, dim=0), torch.cat(c_tmp, dim=0)
        return x, (h_new, c_new)

class Generator(nn.Module):
    def __init__(self,
                 config,
                 tgt_vocab_size):
        super(Generator, self).__init__()
        self.config = config
        self.linear = nn.Linear(config.hidden_size,
                                tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        y = self.softmax(x)
        return y

class Decoder(nn.Module):
    def __init__(self,
                 config,
                 vocab_size):
        super(Decoder, self).__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim + config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.,
            bidirectional=False
        )
        self.attention = Attention(config)
        self.linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(config.dropout)
        self.generator = Generator(config, vocab_size)

    def forward(self, x, h_enc, x_enc, mask=None):
        batch_size = x.size(0)

        xs = self.embedding(x)
        #|xs| = (bs, length, emb_dim)
        h = h_enc
        #|h| = (bs, length, hidden_size)
        prev_output = None

        outs = []
        for i in range(xs.size(1)):
            x = xs[:, i, :].unsqueeze(1)

            if prev_output is None:
                prev_output = torch.zeros(batch_size, 1, self.config.hidden_size)
                #|prev_output| = (bs, 1, hidden_size)

            z, h = self.lstm(torch.cat([x, prev_output], dim=-1), h) #input_feeding: concat 출력과 embedding 결합(softmax에서 발생하는 정보손실 때문)
            # 이전에는 lstm 출력과 embedding을 결합

            c = self.attention(z, x_enc, mask)
            out = self.tanh(self.linear(  # tanh로 정규화
                      torch.cat([z, c], dim=-1)
                      )
                  )

            prev_output = out
            outs += [out]
        result = self.dropout(torch.concat(outs, dim=1))
        y = self.generator(result)

        return y

    def generate(self, h_enc, x_enc, bos_token, mask=None):
        batch_size = x_enc.size(0)
        prev_output = None
        h = h_enc

        x_t = torch.tensor([bos_token] * batch_size).unsqueeze(-1).long()

        result = [bos_token]
        while result[-1] != 2:
            # itos[2] = '<eos>'
            if prev_output is None :
                prev_output = torch.zeros(batch_size, 1, self.config.hidden_size)

            emb_t = self.embedding(x_t)
            z, h = self.lstm(torch.cat([emb_t, prev_output], dim=-1), h)

            c = self.attention(z, x_enc, mask)
            out = self.tanh(self.linear(
                torch.cat([z, c], dim=-1)
            )
            )

            prev_output = out
            y_hat = self.generator(out).argmax(-1)
            # |y_hat| = (bs, 1)
            result.append(y_hat.item())
            x_t = y_hat
        return torch.Tensor(result).long()

class Seq2Seq(nn.Module):
    def __init__(self,
                 config,
                 src_vocab_size,
                 tgt_vocab_size,
                 pad_idx=1):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.pad_idx = pad_idx

        self.encoder = Encoder(config, src_vocab_size)
        self.decoder = Decoder(config, tgt_vocab_size)


    def forward(self, x_enc, x_dec):
        mask = (x_enc == self.pad_idx)
        x_enc, h_enc = self.encoder(x_enc)
        y = self.decoder(x_dec, h_enc, x_enc, mask)

        return y

    def validate(self, x_enc, bos_token):
        with torch.no_grad():
            mask = (x_enc == self.pad_idx)
            x_enc, h_enc = self.encoder(x_enc)
            y = self.decoder.generate(h_enc, x_enc, bos_token, mask)
        return y


class Transformer(nn.Module):
    def __init__(self,
                 config,
                 src_vocab_size,
                 tgt_vocab_size,
                 pad_idx=1,
                 max_len=20):
        super(Transformer, self).__init__()
        self.config = config
        self.pad_idx = pad_idx

        w = torch.exp(-torch.arange(0, config.embedding_dim, 2) * math.log(10000) / config.embedding_dim)
        p = torch.arange(0, max_len).reshape(max_len, 1)
        positional_encoding = torch.zeros((max_len, config.embedding_dim))
        positional_encoding[:, 0 : :2] = torch.sin(p * w)
        positional_encoding[:, 1 : :2] = torch.cos(p * w)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.src_embedding = EncoderPositionalEncoding(config, src_vocab_size, positional_encoding)
        self.tgt_embedding = DecoderPositionalEncoding(config, tgt_vocab_size, positional_encoding)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.n_head,
                dim_feedforward=config.hidden_size,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=config.n_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.embedding_dim,
                nhead=config.n_head,
                dim_feedforward=config.hidden_size,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=config.n_layers
        )
        self.generator = TransformerGenerator(config, tgt_vocab_size)

    def generate_mask(self, src, tgt):
        result = tuple()
        if src is not None:
            src_mask = torch.zeros((src.size(1), src.size(1))).type(torch.bool)
            src_key_padding_mask = (src == self.pad_idx).type(torch.bool)
            result += src_mask, src_key_padding_mask
        if tgt is not None:
            tgt_mask = nn.Transformer().generate_square_subsequent_mask(tgt.size(1))
            tgt_key_padding_mask = (tgt == self.pad_idx).type(torch.bool)
            result += tgt_mask, tgt_key_padding_mask
        return result


    def forward(self, src, tgt):
        src_mask, src_key_padding_mask, tgt_mask, tgt_key_padding_mask = self.generate_mask(src, tgt)

        src_emb, tgt_emb = self.src_embedding(src), self.tgt_embedding(tgt)

        x_enc = self.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        x_dec = self.decoder(
            tgt=tgt_emb,
            memory=x_enc,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        out = self.generator(x_dec)

        return out

class TransformerGenerator(nn.Module):
    def __init__(self,
                 config,
                 tgt_vocab_size):
        super(TransformerGenerator, self).__init__()
        self.config = config
        self.linear = nn.Linear(config.embedding_dim, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        out = self.softmax(x)
        return out

class EncoderPositionalEncoding(nn.Module):
    def __init__(self,
                 config,
                 src_vocab_size,
                 pe,
                 pad_idx=1):
        super(EncoderPositionalEncoding, self).__init__()
        self.config = config
        self.src_embedding = nn.Embedding(src_vocab_size, config.embedding_dim, pad_idx)
        self.positional_encoding = pe
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x_emb = self.src_embedding(x)
        x = x_emb + self.positional_encoding[:, :x_emb.size(1)]
        x = self.dropout(x)
        return x

class DecoderPositionalEncoding(nn.Module):
    def __init__(self,
                 config,
                 tgt_vocab_size,
                 pe,
                 pad_idx=1):
        super(DecoderPositionalEncoding, self).__init__()
        self.config = config
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, config.embedding_dim, pad_idx)
        self.positional_encoding = pe
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x_emb = self.tgt_embedding(x)
        x = x_emb + self.positional_encoding[:, :x_emb.size(1)]
        x = self.dropout(x)
        return x

class Classifier(pl.LightningModule):
    def __init__(self,
                 config,
                 src_vocab_size):
        super(Classifier, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(src_vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.,
            bidirectional=True
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(config.hidden_size * 20, config.hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.accuracy = Accuracy()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.embedding(x)
        x_enc, _ = self.lstm(x)

        x = self.flatten(x_enc)
        x = self.linear2(self.linear1(x))
        y_hat = self.softmax(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = self.embedding(x)
        x_enc, _ = self.lstm(x)

        x = self.flatten(x_enc)
        x = self.linear2(self.linear1(x))
        y_hat = self.softmax(x)
        loss = nn.NLLLoss()(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        return {'loss':loss, 'acc': acc}

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        x = self.embedding(x)
        x_enc, _ = self.lstm(x)

        x = self.flatten(x_enc)
        x = self.linear2(self.linear1(x))
        y_hat = self.softmax(x)
        loss = nn.NLLLoss()(y_hat, y)
        acc = Accuracy()(y_hat, y)
        self.log('valid_loss', loss)
        self.log('valid_acc', acc, on_step=True, on_epoch=False)

