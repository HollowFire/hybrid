# Self-Attention and Dynamic Convolution Hybrid Model for Neural Machine Translation 
This repository contains the source code for the NMT model described in the our paper.
The model is implemented using the [Fairseq](https://github.com/pytorch/fairseq) library. 
In order to reproduce the results in the paper, we recommend cloning this repository to your machine and rebuild.
Note that using a newer version of Fairseq might result in compatibility issues.

## Installation 
* [PyTorch](http://pytorch.org/) version 1.2.0
* Python version 3.7.3

* Install this repository:
```bash
git clone https://github.com/HollowFire/hybrid.git
cd hybrid
pip install --editable ./
```

* Install dynamic convolution module by following the instructions on this [page](https://github.com/pytorch/fairseq/blob/master/examples/pay_less_attention_paper/README.md):
```bash
# to install lightconv
cd fairseq/modules/lightconv_layer
python cuda_function_gen.py
python setup.py install

# to install dynamicconv
cd fairseq/modules/dynamicconv_layer
python cuda_function_gen.py
python setup.py install
```

## Prepare Data
Please follow the instructions in the [Fairseq](https://github.com/pytorch/fairseq) repository to pre-process the datasets. The pre-processed data are stored in the data-bin/ directory.

## Training
Train the model on the iwslt14 De-En dataset:
```bash
SAVE="save/hybrid_iwslt14_de_en"
mkdir -p $SAVE 
python train.py data-bin/iwslt14.tokenized.de-en \
	--user-dir my -a hybrid_iwslt_de_en --save-dir $SAVE \
    --clip-norm 0 --optimizer adam --lr 0.0005 \
    --source-lang de --target-lang en --max-tokens 4000 --no-progress-bar \
    --log-interval 100 --min-lr '1e-09' --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler inverse_sqrt \
    --ddp-backend=no_c10d \
    --max-update 50000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --keep-last-epochs 10 
```

## Test
First Average the last 10 checkpoints:
```bash
python scripts/average_checkpoints.py --inputs $SAVE \
    --num-epoch-checkpoints 10 --output "${SAVE}/checkpoint_last10_avg.pt"
```
Use the averaged  checkpoints to generate on the test set:
```bash
python generate.py data-bin/iwslt14.tokenized.de-en --path "${SAVE}/checkpoint_last10_avg.pt" \
	--batch-size 128 --beam 5 --remove-bpe --lenpen 2.1 --gen-subset test --quiet --user-dir my
```