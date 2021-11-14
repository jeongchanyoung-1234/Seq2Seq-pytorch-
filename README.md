# Sequence-2-Sequence

This model is pytorch-implemented version of Seq2Seq, see the more detailed information below.
https://arxiv.org/abs/1409.3215

## Quick start
### Training
```buildoutcfg
python train.py [--parameters]
```

### Validation
```buildoutcfg
python validate.py [date/addition] [transformer(unstable)/seq2seq] [input]
```
* date: convert input date into uniform style.
  ex)
```buildoutcfg
>python seq2seq/validate.py addition 1+3
4
```
* addition: simply calculate addition. 
  ex)
```buildoutcfg
>python seq2seq/validate.py 7/21/96
1996/07/21
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