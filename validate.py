import re
import sys

import torch

from module.model import Seq2Seq, Transformer

def main(path='C:/Users/JCY/pythonProject1/seq2seq/data/model/') -> str:
    task = sys.argv[1]
    model = sys.argv[2]
    input = sys.argv[3]

    if task == 'date' :
        if model == 'transformer' :
            model_fn = 'seq2seq.date.bs128.ed16.hs128.ep5.transformer.pth'
        elif model == 'seq2seq':
            model_fn = 'seq2seq.date.bs128.ed16.hs128.ep2.pth'
        bos_token = 8
    elif task == 'addition' :
        if model == 'transformer':
            model_fn = 'seq2seq.addition.bs128.ed16.hs128.ep2.transformer.pth'
        elif model == 'seq2seq':
            model_fn = 'seq2seq.addition.bs128.ed8.hs64.ep25.pth'
        bos_token = 3

    dic = torch.load(path + model_fn)
    state_dict = dic['state_dict']
    config = dic['config']
    src_vocab = dic['src_vocab']
    tgt_vocab = dic['tgt_vocab']

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    if model == 'transformer':
        model = Transformer(config, src_vocab_size, tgt_vocab_size)
        model.load_state_dict(state_dict)

    elif model == 'seq2seq':
        model = Seq2Seq(config, src_vocab_size, tgt_vocab_size)
        model.load_state_dict(state_dict)

    input = list(re.sub(' {1,}', ' ', input).lower().strip())

    id = []
    for i in range(len(input)) :
        id.append(src_vocab.stoi[input[i]])
    id = torch.Tensor(id).long().unsqueeze(0)

    if config.reverse_input :
        idx = torch.arange(id.size(1) - 1, -1, -1).long()
        id = torch.index_select(id, 1, idx)
    y_hat = list(model.validate(id, bos_token=bos_token))

    result = []
    for i in range(len(y_hat)) :
        idx = y_hat[i]
        result.append(tgt_vocab.itos[idx])
    output = ''.join(result[1 :-1])

    return output

if __name__ == '__main__':
    output = main()
    print(output)




