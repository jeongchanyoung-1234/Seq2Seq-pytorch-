import math

import torch
import torch.nn as nn


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
        for i in range(xs.size(1) - 1):
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


class TransformerGenerator(nn.Module):
    def __init__(self,
                 config,
                 tgt_vocab_size):
        super(TransformerGenerator, self).__init__()
        self.linear = nn.Linear(config.embedding_dim, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


class Transformer(nn.Module): # embedding, encoding, generator(emb, vocab) 정의 필요
    def __init__(self,
                 config,
                 src_vocab_size,
                 tgt_vocab_size,
                 pad_idx=1):
        super(Transformer, self).__init__()
        self.config = config
        self.pad_idx = pad_idx
        self.model = nn.Transformer(
            d_model=config.embedding_dim,
            nhead=config.n_head,
            num_encoder_layers=config.n_layers,
            num_decoder_layers=config.n_layers,
            dim_feedforward=config.hidden_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.src_embedding = nn.Embedding(src_vocab_size, config.embedding_dim, pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, config.embedding_dim, pad_idx)

        w = torch.exp(-torch.arange(0, config.embedding_dim, 2) * math.log(10000) / config.embedding_dim)
        p = torch.arange(0, 10).unsqueeze(-1)
        self.positional_encoding = torch.zeros((10, config.embedding_dim))
        self.positional_encoding[:, 0::2] = torch.sin(w * p)
        self.positional_encoding[:, 1::2] = torch.cos(w * p)

        self.generator = TransformerGenerator(config, tgt_vocab_size)

    def generate_mask(self, src, tgt=None) -> tuple:
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))

        src_key_padding_mask = (src == self.pad_idx).type(torch.bool)
        tgt_key_padding_mask = (tgt == self.pad_idx).type(torch.bool)

        return src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask

    def forward(self, src, tgt):
        _, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.generate_mask(src, tgt)
        src, tgt = self.src_embedding(src) + self.positional_encoding[:src.size(1), :], self.tgt_embedding(tgt) + self.positional_encoding[:tgt.size(1), :]
        out = self.model(
            src, tgt,
            src_mask=None,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        out = self.generator(out)[:, 1:, :] # sequential하지 않기 때문에 bos를 같이 반환
        return out

    def validate(self, src, bos_token):
        src_key_padding_mask = (src == self.pad_idx).type(torch.bool)
        src = self.src_embedding(src) + self.positional_encoding[:src.size(1), :]
        memory = self.model.encoder(
            src,
            mask=None,
            src_key_padding_mask=src_key_padding_mask
        )

        result = [bos_token]
        while result[-1] != 2:
            tgt = torch.Tensor(result).type(torch.int).unsqueeze(0)
            tgt_key_padding_mask = (tgt == self.pad_idx).type(torch.bool)
            tgt = self.tgt_embedding(tgt) + self.positional_encoding[:tgt.size(1)]
            tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(1))
            y_hat = self.model.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            ).argmax(-1)[:, -1].item()
            result.append(y_hat)

if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.hidden_size = 32
            self.n_layers = 2
            self.dropout = .5

    config = Config()
    h_enc = torch.rand(16, 10, 32)
    h_t_dec = torch.rand(16, 1, 32)

    attention = Attention()
    attention(h_t_dec, h_enc)