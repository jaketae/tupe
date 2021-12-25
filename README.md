# TUPE

PyTorch implementation of [Rethinking Positional Encoding in Language Pre-training](https://arxiv.org/abs/2006.15595).

![alt text](https://miro.medium.com/max/1200/1*TEG19rbIcY3znMBTpTrNjA.png)

## Quickstart

Clone this repository.

```sh
git clone https://github.com/jaketae/tupe.git
```

 Navigate to the cloned directory. You can use the bare-bone TUPE Encoder model via

```python
>>> import torch; from tupe import TUPEConfig, TUPEEncoder
>>> config  = TUPEConfig()
>>> model = TUPEEncoder(config)
>>> x = torch.randn(8, 100, 128)
>>> model(x).shape
torch.Size([8, 100, 128])
```

By default, the model comes with the following parameters:

```python
TUPEConfig(
    num_layers=6, 
    num_heads=8, 
    d_model=128, 
    d_head=16, 
    max_len=256, 
    dropout=0.1, 
    expansion_factor=1, 
    relative_bias=True, 
    bidirectional_bias=True, 
    num_buckets=32, 
    max_distance=128
)
```

## Abstract

> In this work, we investigate the positional encoding methods used in language pre- training (e.g., BERT) and identify several problems in the existing formulations. First, we show that in the absolute positional encoding, the addition operation applied on positional embeddings and word embeddings brings mixed correlations between the two heterogeneous information resources. It may bring unnecessary randomness in the attention and further limit the expressiveness of the model. Sec- ond, we question whether treating the position of the symbol [CLS] the same as other words is a reasonable design, considering its special role (the representation of the entire sentence) in the downstream tasks. Motivated from above analysis, we propose a new positional encoding method called Transformer with Untied Positional Encoding (TUPE). In the self-attention module, TUPE computes the word contextual correlation and positional correlation separately with different parameterizations and then adds them together. This design removes the mixed and noisy correlations over heterogeneous embeddings and offers more expres- siveness by using different projection matrices. Furthermore, TUPE unties the [CLS] symbol from other positions, making it easier to capture information from all positions. Extensive experiments and ablation studies on GLUE benchmark demonstrate the effectiveness of the proposed method.

## Implementation Notes

* The default configuration follows TUPE-R, which includes T5's relative position bias. To use TUPE-A, simply toggle `TUPEConfig.relative_bias` field to `False`.
* To avoid limiting the use case of this architecture to BERT-type models with `[CLS]` tokens, this implementation purposefully omits Section 3.2, on untying the `[CLS]` symbol from positions.

## Citation

```bibtex
@inproceedings{ke2021rethinking,
	title        = {Rethinking Positional Encoding in Language Pre-training},
	author       = {Guolin Ke and Di He and Tie-Yan Liu},
	year         = 2021,
	booktitle    = {International Conference on Learning Representations},
	url          = {https://openreview.net/forum?id=09-528y2Fgf}
}
```

