<div align="center">
<h1 align="center">☀️BRIGHT☀️</h1>

<h3>BRIGHT: A globally distributed multimodal VHR dataset for all-weather disaster response</h3>


[Hongruixuan Chen](https://scholar.google.ch/citations?user=XOk4Cf0AAAAJ&hl=zh-CN&oi=ao)<sup>1,2</sup>, [Jian Song](https://scholar.google.ch/citations?user=CgcMFJsAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Olivier Dietrich](https://scholar.google.ch/citations?user=st6IqcsAAAAJ&hl=de)<sup>3</sup>, [Clifford Broni-Bediako](https://scholar.google.co.jp/citations?user=Ng45cnYAAAAJ&hl=en)<sup>2</sup>, [Weihao Xuan](https://scholar.google.com/citations?user=7e0W-2AAAAAJ&hl=en)<sup>1,2</sup>, [Junjue Wang](https://scholar.google.com.hk/citations?user=H58gKSAAAAAJ&hl=en)<sup>1</sup>  
[Xinlei Shao](https://scholar.google.com/citations?user=GaRXJFcAAAAJ&hl=en)<sup>1</sup>, [Yimin Wei](https://www.researchgate.net/profile/Yimin-Wei-9)<sup>1,2</sup>, [Junshi Xia](https://scholar.google.com/citations?user=n1aKdTkAAAAJ&hl=en)<sup>3</sup>, [Cuiling Lan](https://scholar.google.com/citations?user=XZugqiwAAAAJ&hl=zh-CN)<sup>4</sup>, [Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en)<sup>3</sup>, [Naoto Yokoya](https://scholar.google.co.jp/citations?user=DJ2KOn8AAAAJ&hl=en)<sup>1,2 *</sup>


<sup>1</sup> The University of Tokyo, <sup>2</sup> RIKEN AIP,  <sup>3</sup> ETH Zurich,  <sup>4</sup> Microsoft Research Asia

[![ESSD paper](https://img.shields.io/badge/ESSD-paper-cyan)](https://essd.copernicus.org/preprints/essd-2025-269/) [![arXiv paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2501.06019) [![Zenodo Dataset](https://img.shields.io/badge/Zenodo-Dataset-blue)](https://zenodo.org/records/14619797)   [![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Kullervo/BRIGHT) [![Zenodo Model](https://img.shields.io/badge/Zenodo-Model-green)](https://zenodo.org/records/15349462) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=ChenHongruixuan.BRIGHT&left_color=%2363C7E6&right_color=%23CEE75F)


[**Overview**](#overview) | [**Start BRIGHT**](#%EF%B8%8Flets-get-started-with-bright) | [**Common Issues**](#common-issues) | [**Others**](#q--a) 


</div>

## 🛎️Updates
* **` Notice☀️☀️`**: The [full version of the BRIGHT paper](https://arxiv.org/abs/2501.06019) is now online!! The contents related to IEEE GRSS DFC 2025 have been transferred to [here](bda_benchmark/README_DFC25.md)!!
* **` Oct 17th, 2025`**: BRIGHT has been accepted by [ESSD](https://essd.copernicus.org/preprints/essd-2025-269/)!!
* **` Aug 12th, 2025`**: BRIGHT has been integrated into [TorChange](https://github.com/Z-Zheng/pytorch-change-models). Many thanks for the effort of [Dr. Zhuo Zheng](https://zhuozheng.top/)!!
* **` May 05th, 2025`**: All the data and benchmark code related to our paper has now released. You are warmly welcome to use them!!
* **` Apr 28th, 2025`**: IEEE GRSS DFC 2025 Track II is over. Congratulations to [winners](https://www.grss-ieee.org/community/technical-committees/winners-of-the-2025-ieee-grss-data-fusion-contest-all-weather-land-cover-and-building-damage-mapping/)!! You can now download the full version of DFC 2025 Track II data in [Zenodo](https://zenodo.org/records/14619797) or [HuggingFace](https://huggingface.co/datasets/Kullervo/BRIGHT)!! 
* **` Jan 18th, 2025`**: BRIGHT has been integrated into [TorchGeo](https://github.com/microsoft/torchgeo). Many thanks for the effort of [Nils Lehmann](https://github.com/nilsleh)!!
* **` Jan 13th, 2025`**: The [arXiv paper](https://arxiv.org/abs/2501.06019) of BRIGHT is now online. If you are interested in details of BRIGHT, do not hesitate to take a look!!

## 🔭Overview

* [**BRIGHT**](https://arxiv.org/abs/2501.06019) is the first open-access, globally distributed, event-diverse multimodal dataset specifically curated to support AI-based disaster response. It covers **five** types of natural disasters and **two** types of man-made disasters across **14** disaster events in **23** regions worldwide, with a particular focus on developing countries. 


* It supports not only the development of **supervised** deep models, but also the testing of their performance on **cross-event transfer** setup, as well as **unsupervised domain adaptation**, **semi-supervised learning**, **unsupervised change detection**, and **unsupervised image matching** methods in multimodal and disaster scenarios.

<p align="center">
  <img src="./figure/overall.jpg" alt="accuracy" width="97%">
</p>




## 🗝️Let's Get Started with BRIGHT!
### `A. Installation`

Note that the code in this repo runs under **Linux** system. We have not tested whether it works under other OS.

**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/ChenHongruixuan/BRIGHT.git
cd BRIGHT
```

**Step 2: Environment Setup:**

It is recommended to set up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n bright-benchmark
conda activate bright-benchmark
```

***Install dependencies***

```bash
pip install -r requirements.txt
```



### `B. Data Preparation`
Please download the BRIGHT from [Zenodo](https://zenodo.org/records/14619797) or [HuggingFace](https://huggingface.co/datasets/Kullervo/BRIGHT). Note that we cannot redistribute the optical data over Ukraine, Myanmar, and Mexico. Please follow our [tutorial](./tutorial.md) to download and preprocess them. 

After the data has been prepared, please make them have the following folder/file structure:
```
${DATASET_ROOT}   # Dataset root directory, for example: /home/username/data/bright
│
├── pre-event
│    ├──bata-explosion_00000000_pre_disaster.tif
│    ├──bata-explosion_00000001_pre_disaster.tif
│    ├──bata-explosion_00000002_pre_disaster.tif
│   ...
│
├── post-event
│    ├──bata-explosion_00000000_post_disaster.tif
│    ... 
│
└── target
     ├──bata-explosion_00000000_building_damage.tif 
     ...   
```

### `C. Model Training & Tuning`

The following commands show how to train and evaluate UNet on the BRIGHT dataset using our standard ML split set in [`bda_benchmark/dataset/splitname/standard_ML`]:

```bash
python script/standard_ML/train_UNet.py --dataset 'BRIGHT' \
                                        --train_batch_size 16 \
                                        --eval_batch_size 4 \
                                        --num_workers 16 \
                                        --crop_size 640 \
                                        --max_iters 800000 \
                                        --learning_rate 1e-4 \
                                        --model_type 'UNet' \
                                        --model_param_path '<your model checkpoint saved path>' \
                                        --train_dataset_path '<your dataset path>' \
                                        --train_data_list_path '<your project path>/bda_benchmark/dataset/splitname/standard_ML/train_set.txt' \
                                        --val_dataset_path '<your dataset path>' \
                                        --val_data_list_path '<your project path>/bda_benchmark/dataset/splitname/standard_ML/val_set.txt' \
                                        --test_dataset_path '<your dataset path>' \
                                        --test_data_list_path '<your project path>/bda_benchmark/dataset/splitname/standard_ML/test_set.txt' 
```


### `D. Inference & Evaluation`
Then, you can run the following code to generate raw & visualized prediction results and evaluate performance using the saved weight. You can also download our provided checkpoints from [Zenodo](https://zenodo.org/records/15349462).

```bash
python script/standard_ML/infer_UNet.py --model_path  '<path of the checkpoint of model>' \
                                        --test_dataset_path '<your dataset path>' \
                                        --test_data_list_path '<your project path>/bda_benchmark/dataset/splitname/standard_ML/test_set.txt' \
                                        --output_dir '<your inference results saved path>'
```

### `E. Other Benchmarks & Setup`
In addition to the above supervised deep models, BRIGHT also provides standardized evaluation setups for several important learning paradigms and multimodal EO tasks:

* [`Cross-event transfer setup`](bda_benchmark/README_cross_event.md): Evaluate model generalization across disaster types and regions. This setup simulates real-world scenarios where no labeled data (**zero-shot**) or limited labeled data (**one-shot**) is available for the target event during training. 

* [`Unsupervised domain adaptation`](bda_benchmark/README_cross_event.md): Adapt models trained on source disaster events to unseen target events without any target labels, using UDA techniques under the **zero-shot** cross-event setting.

* [`Semi-supervised learning`](bda_benchmark/README_cross_event.md): Leverage a small number of labeled samples and a larger set of unlabeled samples from the target event to improve performance under the **one-shot** cross-event setting.

* [`Unsupervised multimodal change detection`](umcd_benchmark/README.md): Detect disaster-induced building changes without using any labels. This setup supports benchmarking of general-purpose change detection algorithms under realistic large-scale disaster scenarios.

* [`Unsupervised multimodal image matching`](umim_benchmark/README.md): Evaluate the performance of matching algorithms in aligning **raw, large-scale** optical and SAR images based on **manual-control-point**-based registration accuracy. This setup focuses on realistic multimodal alignment in disaster-affected areas.

* [`IEEE GRSS DFC 2025 Track II`](bda_benchmark/README_DFC25.md): The Track II of [IEEE GRSS DFC 2025](https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/) aims to develop robust and generalizable methods for assessing building damage using bi-temporal multimodal images on unseen disaster events.



## 🤔Common Issues
Based on peers' questions from [issue section](https://github.com/ChenHongruixuan/BRIGHT/issues), here's a quick navigate list of solutions to some common issues.

| Issue | Solution | 
| :---: | :---: | 
|  Complete data of DFC25 for research |   The labels for validation and test sets of DFC25 have been uploaded to [Zenodo](https://zenodo.org/records/14619797) and [HuggingFace](https://huggingface.co/datasets/Kullervo/BRIGHT).     |
|  Python package conflicts   |   The baseline code is not limited to a specific version, and participants do not need to match the version we provide.     |

<!-- ## 📜Follow-Up Works
| Paper | Publication date | Venue | Link |
| :-- | :--: | :-- | :-- | -->



## 📜Reference

If this dataset or code contributes to your research, please kindly consider citing our paper and give this repo ⭐️ :)
```
@article{chen2025bright,
      title={BRIGHT: A globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response}, 
      author={Hongruixuan Chen and Jian Song and Olivier Dietrich and Clifford Broni-Bediako and Weihao Xuan and Junjue Wang and Xinlei Shao and Yimin Wei and Junshi Xia and Cuiling Lan and Konrad Schindler and Naoto Yokoya},
      journal={arXiv preprint arXiv:2501.06019},
      year={2025},
      url={https://arxiv.org/abs/2501.06019}, 
}
```

## 🤝Acknowledgments
The authors would also like to give special thanks to [Sarah Preston](https://www.linkedin.com/in/sarahjpreston/) of Capella Space, [Capella Space's Open Data Gallery](https://www.capellaspace.com/earth-observation/gallery), [Maxar Open Data Program](https://www.maxar.com/open-data) and [Umbra Space's Open Data Program](https://umbra.space/open-data/) for providing the valuable data.

## 🙋Q & A
***For any questions, please feel free to leave it in the [issue section](https://github.com/ChenHongruixuan/BRIGHT/issues) or [contact us.](mailto:Qschrx@gmail.com)***
