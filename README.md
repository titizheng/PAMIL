# Dynamic Policy-Driven Adaptive Multi-Instance Learning for Whole Slide Image Classification


Tingting Zheng,  [Kui jiang](https://scholar.google.com/citations?user=AbOLE9QAAAAJ&hl=en&oi=ao), [Hongxun Yao](https://scholar.google.com/citations?user=aOMFNFsAAAAJ)

Harbin Institute of Technology
## Update
- [2024/02/29] The repo is created
- [2024/07/27] Update preprocessing features



## Usage
  ### Dataset

   #### Preprocess TCGA Dataset

>We use the same configuration of data preprocessing as [DSMIL](https://github.com/binli123/dsmil-wsi).

   #### Preprocess CAMELYON16 Dataset

>We use [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) to preprocess CAMELYON16 at 20x.

   #### Preprocessed feature vector

> We provide processed feature vector for two datasets. Aforementioned works [DSMIL](https://github.com/binli123/dsmil-wsi) and [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) 
greatly simplified the preprocessing. Thanks again to their wonderful works!
>We use preprocessing features from [MMIL](https://github.com/hustvl/MMIL-Transformer?tab=readme-ov-file). More details about this file can refer [DSMIL](https://github.com/binli123/dsmil-wsi) and [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) Thanks to their wonderful works!

<div align="center">
  
| Dataset | Link 
|------------|:-----:|----|
| `TCGA`|[HF link](https://pan.quark.cn/s/b6c014c29528)
| `CAMELYON16-Testing`|[HF link](https://pan.quark.cn/s/7339cfb8c26c)
| `CAMELYON16-Training and validation`|[HF link](https://pan.quark.cn/s/27f392595e83)
</div>


## Cite this work

```
@inproceedings{zheng2024dynamic,
    title={Dynamic Policy-Driven Adaptive Multi-Instance Learning for Whole Slide Image Classification},
    author={Zheng, Tingting and
            Jiang, Kui and
            Yao, Hongxun},
    booktitle={CVPR},
    year={2024}
}
```

