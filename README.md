# Sequence-2-Sequence

This model is pytorch-implemented version of Seq2Seq, focusing on addition and date formation problem. See the more detailed information below.
https://arxiv.org/abs/1409.3215

## Quick start
### Training
```buildoutcfg
>python main.py april/10/2001
  날짜 변환 문제네요.
  제 생각에 이 날짜는 2001/04/10라고 표현할 수 있을 것 같습니다.
```

```buildoutcfg
>python main.py 11+75
  간단한 덧셈 문제네요.
  제 생각에 정답은 86인 것 같습니다.
```

## Training Parameters
* batch_size: the size of mini-batches (default 128)
* hidden_size: the size of hidden states (defualt 128)
* embedding_dim: the output size of embedding layer (default 16)
* n_layers: the number of layers in both encoder and decoder (default 2)
* dropout: the ratio of nodes dropped (default 0.5)
* optimizer: Adam/SGD/RMSprop supported (default adam)
* lr: the learning rate (default 0.01)
* max_grad_norm: gradient clipping (default 5.0)
* epochs: the number of iteration training the whole batches (default 5)
* verbose: print training logs every n-epochs (default 1)
* reverse_input: reversing the input, which is suggested at GNMT (True if provided)
* use_transformer: use transformer (unstable)
* file_fn: training source filename