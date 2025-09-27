# Waterlogformer: A Multimodal Model for Waterlogging

## Introduction

Waterlogformer is a multimodal model for WD prediction.

To model ***hydrological mechanisms*** and effectively fuse ***multimodal data***, a dual-branch multimodal architecture is developed for WD prediction, comprising three key components:

* **Rainfall Branch** — employs a Terrain-aware Rainfall Accumulation Unit which simulates rainfall accumulation over time and across locations under specific terrain conditions, embedding hydrodynamic knowledge of how rain propagates over landscapes.
* **Waterlogging Branch** — leverages historical WD time series together with static geographical information to capture spatio-temporal waterlogging patterns while respecting geographic constraints.
* **Multimodal Fusion Prediction Module** — integrates rainfall and historical WD representations and incorporates a distance- and terrain-similarity–based contrastive learning mechanism to enhance sensitivity to critical geographical factors during multimodal fusion.

Experiment results on a real-world dataset demonstrate the superior performance of Waterlogformer.


![1758953951295](https://file+.vscode-resource.vscode-cdn.net/Users/cheney/Documents/02025%20%E5%8D%8E%E4%B8%9C%E5%B8%88%E8%8C%83%E5%A4%A7%E5%AD%A6/%E7%A0%94%E9%9B%B6%202024-2025/Waterlogformer/image/README/1758953951295.png)



## Prerequisites

Before proceeding, ensure Python 3.9 is installed. Install the required dependencies with the following command:

```
pip install -r requirements.txt
```

## Acknowledgements

Our gratitude extends to the authors of the following repositories for their foundational model implementations:

- [Autoformer](https://github.com/thuml/Autoformer)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [Crossformer](https://github.com/Thinklab-SJTU/Crossformer)
