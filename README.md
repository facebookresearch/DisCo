# DisCo Transformer for Non-autoregressive MT


### Download trained DisCo transformers
All models are trained with distillation. See our [paper](https://arxiv.org/abs/2001.05136).
Pretrained Models | | | |
|---|---|---|---|
[WMT14 English-German](https://drive.google.com/uc?id=1llsS8XMVoZx0E5z0Es9OOkGcN3GYi_t4) | [WMT14 German-English](https://drive.google.com/uc?id=16a0Sul8yytxIeM7eA3kOO0VJEPJNhbA0) |  [WMT16 English-Romanian](https://drive.google.com/uc?id=158HQhpglOpN-zmDe8Q-c7xw4RxUwNLpI) |  [WMT16 Romanian-English](https://drive.google.com/uc?id=1olRGAL8UVIhnNks2N9Up2EDP01I-jt8Q)
[WMT17 English-Chinese](https://drive.google.com/uc?id=1mAD-SBJToQpQ1Tv760cBMGCZYUIIVoPQ) | [WMT17 Chinese-English](https://drive.google.com/uc?id=1GbMgiTGoDqdpM18Tn-SUf7-x7_IXqK0G) | [WMT14 English-French](https://drive.google.com/uc?id=1nZetAse2DNHrGZKrn-d00o8QeZOlgURp)

We also provide our knowledge distillation data for [WMT14 EN-DE](https://drive.google.com/a/cs.washington.edu/uc?id=12sYzTqj-nAsi0ky65-Sn3tY6lmv6h8PP).


### Preprocess

```bash
text=PATH_YOUR_DATA

output_dir=PATH_YOUR_OUTPUT

src=source_language

tgt=target_language

model_path=PATH_TO_MASKPREDICT_MODEL_DIR

python preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref $text/train \
    --validpref $text/valid --testpref $text/test  --destdir ${output_dir}/data-bin \
    --workers 60 --srcdict ${model_path}/maskPredict_${src}_${tgt}/dict.${src}.txt \
    --tgtdict ${model_path}/maskPredict_${src}_${tgt}/dict.${tgt}.txt
```

### Train


```bash
model_dir=PLACE_TO_SAVE_YOUR_MODEL

python train.py ${output_dir}/data-bin --arch disco_transformer \
    --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 --lr 5e-4 \
    --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_self \
    --max-tokens 8192 --weight-decay 0.01 --dropout 0.2 --encoder-layers 6 --encoder-embed-dim 512 \
    --decoder-layers 6 --decoder-embed-dim 512 --fp16 --max-source-positions 10000 \
    --max-target-positions 10000 --max-update 300000 --seed 1 \
    --save-dir ${model_dir} --dynamic-masking  --ignore-eos-loss --share-all-embeddings
```

### Evaluation
We provide two inference methods:
1. Parallel Easy-First ([Kasai et al., 2020](https://arxiv.org/abs/2001.05136))
```bash
python generate_disco.py ${output_dir}/data-bin --path ${model_dir}/checkpoint_best.pt \
    --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 10 \
    --decoding-strategy easy_first --length-beam 5
```
2. Mask-Predict ([Ghazvininejad et al., 2019](https://arxiv.org/abs/1904.09324))
```bash
python generate_disco.py ${output_dir}/data-bin --path ${model_dir}/checkpoint_best.pt \
    --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 10 \
    --decoding-strategy mask_predict --length-beam 5
```

# License
DisCo is CC-BY-NC 4.0.
The license applies to the trained models as well.

# Citation

Please cite as:

```bibtex
@inproceedings{Kasai2020DisCo,
  title = {Non-autoregressive Machine Translation with Disentangled Context Transformer},
  author = {Jungo Kasai and James Cross and Marjan Ghazvininejad and Jiatao Gu},
  booktitle = {Proc. of ICML},
  year = {2020},
  url = {https://arxiv.org/abs/2001.05136},
}
```

## Note

We based this code heavily on the original [mask-predict implementation](https://github.com/facebookresearch/Mask-Predict).
