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
>For your own dataset, you can modify and run [create_patches_fp_Lung.py](https://github.com/titizheng/PAMIL/blob/main/slide_preproce/create_patches_fp_Lung.py) and [extract_features_fp_LungRes18Imag.py](https://github.com/titizheng/PAMIL/blob/main/slide_preproce/extract_features_fp_LungRes18Imag.py). 


   #### Preprocessed feature vector

>We use preprocessing features from [MMIL](https://github.com/hustvl/MMIL-Transformer?tab=readme-ov-file). More details about this file can refer [DSMIL](https://github.com/binli123/dsmil-wsi) and [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) Thanks to their wonderful works!

<div align="center">
  
| Dataset | Link | 
|------------|:-----|
| `TCGA`|[HF link](https://pan.quark.cn/s/e3d80653294d)
| `CAMELYON16-Testing`|[HF link](https://pan.quark.cn/s/0000a04c9895)
| `CAMELYON16-Training and validation`|[HF link](https://pan.quark.cn/s/0877b64faf78)
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

