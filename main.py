import re
import sys

import torch
from module.model import Classifier, Seq2Seq
from module.data_loader import ClsDataLoader

class Config:
    def __init__(self):
        self.file_fn = 'classify.csv'
        self.batch_size = 256

def get_task_id():
    dataloader = ClsDataLoader(Config())
    dataloader.get_dataloader()

    model = Classifier.load_from_checkpoint(
        'C:/Users/JCY/pythonProject1/seq2seq/data/model/cls/epoch=01-train_acc=1.00.ckpt'
    )

    model.eval()
    input = sys.argv[1]
    result = []
    for i in range(10):
        try:
            char = list(input)[i]
            result.append(dataloader.src.vocab.stoi[char])
        except:
            result.append(1)

    y_hat = model(torch.Tensor(result).long().unsqueeze(0)).argmax(-1).item()

    return y_hat

def main(task_id, path='C:/Users/JCY/pythonProject1/seq2seq/data/model/') -> str:
    if task_id == 1 :
        model_fn = 'seq2seq.date.bs128.ed16.hs128.ep2.pth'
        bos_token = 8
    elif task_id == 0:
        model_fn = 'seq2seq.addition.bs128.ed8.hs64.ep25.pth'
        bos_token = 3

    dic = torch.load(path + model_fn)
    state_dict = dic['state_dict']
    config = dic['config']
    src_vocab = dic['src_vocab']
    tgt_vocab = dic['tgt_vocab']

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    model = Seq2Seq(config, src_vocab_size, tgt_vocab_size)
    model.load_state_dict(state_dict)

    input = list(re.sub(' {1,}', ' ', sys.argv[1]).lower().strip())

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
    task_id = get_task_id()
    output = main(task_id)
    if task_id == 0:
        print('간단한 덧셈 문제네요.')
        print('제 생각에 정답은 {}인 것 같습니다.'.format(output))
    elif task_id == 1:
        print('날짜 변환 문제네요.')
        print('제 생각에 이 날짜는 {}라고 표현할 수 있을 것 같습니다.'.format(output))


