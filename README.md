# Waterlogformer: A Multimodal Model for Waterlogging Prediction

**[WSDM 2026]** This paper has been accepted by the 19th ACM International Conference on Web Search and Data Mining (WSDM 2026).

## Introduction

Waterlogformer is a multimodal model for WD prediction.

To model ***hydrological mechanisms*** and effectively fuse ***multimodal data***, a dual-branch multimodal architecture is developed for WD prediction, comprising three key components:

* **Rainfall Branch** — employs a Terrain-aware Rainfall Accumulation Unit which simulates rainfall accumulation over time and across locations under specific terrain conditions, embedding hydrodynamic knowledge of how rain propagates over landscapes.
* **Waterlogging Branch** — leverages historical WD time series together with static geographical information to capture spatio-temporal waterlogging patterns while respecting geographic constraints.
* **Multimodal Fusion Prediction Module** — integrates rainfall and historical WD representations and incorporates a distance- and terrain-similarity–based contrastive learning mechanism to enhance sensitivity to critical geographical factors during multimodal fusion.

Experiment results on a real-world dataset demonstrate the superior performance of Waterlogformer.

![1758953951295](./image/README/1758953951295.png)

## Prerequisites

Before proceeding, ensure Python 3.9 is installed. Install the required dependencies with the following command:

```
pip install -r requirements.txt
```

## Acknowledgements

Our gratitude extends to the authors of the following repositories for their foundational model implementations:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [Autoformer](https://github.com/thuml/Autoformer)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [Crossformer](https://github.com/Thinklab-SJTU/Crossformer)
