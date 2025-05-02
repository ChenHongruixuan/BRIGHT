<div align="center">
<h1 align="center">☀️BRIGHT☀️</h1>

<h3>BRIGHT: A globally distributed multimodal VHR dataset for all-weather disaster response</h3>


[Hongruixuan Chen](https://scholar.google.ch/citations?user=XOk4Cf0AAAAJ&hl=zh-CN&oi=ao)<sup>1,2</sup>, [Jian Song](https://scholar.google.ch/citations?user=CgcMFJsAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Olivier Dietrich](https://scholar.google.ch/citations?user=st6IqcsAAAAJ&hl=de)<sup>3</sup>, [Clifford Broni-Bediako](https://scholar.google.co.jp/citations?user=Ng45cnYAAAAJ&hl=en)<sup>2</sup>, [Weihao Xuan](https://scholar.google.com/citations?user=7e0W-2AAAAAJ&hl=en)<sup>1,2</sup>, [Junjue Wang](https://scholar.google.com.hk/citations?user=H58gKSAAAAAJ&hl=en)<sup>1</sup>  
[Xinlei Shao](https://scholar.google.com/citations?user=GaRXJFcAAAAJ&hl=en)<sup>1</sup>, [Yimin Wei](https://www.researchgate.net/profile/Yimin-Wei-9)<sup>1,2</sup>, [Junshi Xia](https://scholar.google.com/citations?user=n1aKdTkAAAAJ&hl=en)<sup>3</sup>, [Cuiling Lan](https://scholar.google.com/citations?user=XZugqiwAAAAJ&hl=zh-CN)<sup>4</sup>, [Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en)<sup>3</sup>, [Naoto Yokoya](https://scholar.google.co.jp/citations?user=DJ2KOn8AAAAJ&hl=en)<sup>1,2 *</sup>


<sup>1</sup> The University of Tokyo, <sup>2</sup> RIKEN AIP,  <sup>3</sup> ETH Zurich,  <sup>4</sup> Microsoft Research Asia

[![arXiv paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2501.06019)  [![Codalab Leaderboard](https://img.shields.io/badge/Codalab-Leaderboard-cyan)](https://codalab.lisn.upsaclay.fr/competitions/21122) [![Zenodo Dataset](https://img.shields.io/badge/Zenodo-Dataset-blue)](https://zenodo.org/records/15322113)   [![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Kullervo/BRIGHT) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=ChenHongruixuan.BRIGHT&left_color=%2363C7E6&right_color=%23CEE75F)


[**Overview**](#overview) | [**Start BRIGHT**](#%EF%B8%8Flets-get-started-with-bright) | [**Common Issues**](#common-issues) | [**Others**](#q--a) 


</div>

## 🛎️Updates
* **` Notice☀️☀️`**: The [full version of the BRIGHT paper](https://arxiv.org/abs/2501.06019) are now online. Related data and benchmark suites will be released soon!!
* **` Apr 28th, 2025`**: IEEE GRSS DFC 2025 Track II is over. Congratulations to [winners](https://www.grss-ieee.org/community/technical-committees/winners-of-the-2025-ieee-grss-data-fusion-contest-all-weather-land-cover-and-building-damage-mapping/)!! You can now download the full version of DFC 2025 Track II data in [Zenodo](https://zenodo.org/records/15322113) or [HuggingFace](https://huggingface.co/datasets/Kullervo/BRIGHT)!!
* **` Jan 18th, 2025`**: BRIGHT has been integrated into [TorchGeo](https://github.com/microsoft/torchgeo). Many thanks for the effort of [Nils Lehmann](https://github.com/nilsleh)!!
* **` Jan 13th, 2025`**: The [arXiv paper](https://arxiv.org/abs/2501.06019) of BRIGHT is now online. If you are interested in details of BRIGHT, do not hesitate to take a look!!
* **` Jan 13th, 2025`**: The benchmark code for IEEE GRSS DFC 2025 Track II is now available. Please follow the [**instruction**](#%EF%B8%8Flets-get-started-with-dfc-2025) to use it!! Also, you can find dataset and code related to Track I in [here](https://github.com/cliffbb/DFC2025-OEM-SAR-Baseline)!! 

## 🔭Overview

* [**BRIGHT**](https://arxiv.org/abs/2501.06019) is the first open-access, globally distributed, event-diverse multimodal dataset specifically curated to support AI-based disaster response. It covers **five** types of natural disasters and **two** types of man-made disasters across **14** disaster events in **23** regions worldwide, with a particular focus on developing countries. 


* It supports not only the development of *supervised deep models*, but also the testing of their performance on *cross-event transfer* setup, as well as *unsupervised domain adaptation*, *semi-supervised learning*, *unsupervised change detection*, and *unsupervised image matching* methods in multimodal and disaster scenarios.

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
Please download the BRIGHT from [Zenodo](https://zenodo.org/records/14950271) or [HuggingFace](https://huggingface.co/datasets/Kullervo/BRIGHT) and make them have the following folder/file structure:
```
${DATASET_ROOT}   # Dataset root directory, for example: /home/username/data/dfc25_track2_trainval
│
├── train
│    ├── pre-event
│    │    ├──bata-explosion_00000000_pre_disaster.tif
│    │    ├──bata-explosion_00000001_pre_disaster.tif
│    │    ├──bata-explosion_00000002_pre_disaster.tif
│    │   ...
│    │
│    ├── post-event
│    │    ├──bata-explosion_00000000_post_disaster.tif
│    │    ... 
│    │
│    └── target
│         ├──bata-explosion_00000000_building_damage.tif 
│         ...   
│   
└── val
     ├── pre-event
     │    ├──bata-explosion_00000003_pre_disaster.tif
     │   ...
     │
     └── post-event
          ├──bata-explosion_00000003_post_disaster.tif
         ...
```

### `C. Model Training & Tuning`

The following commands show how to train and evaluate UNet on the BRIGHT dataset using our standard ML split set in [`dfc25_benchmark/dataset/splitname`]:

```bash
python script/train_baseline_network.py  --dataset 'BRIGHT' \
                                          --train_batch_size 16 \
                                          --eval_batch_size 4 \
                                          --num_workers 1 \
                                          --crop_size 640 \
                                          --max_iters 800000 \
                                          --learning_rate 1e-4 \
                                          --model_type 'UNet' \
                                          --train_dataset_path '<your dataset path>/train' \
                                          --train_data_list_path '<your project path>/dfc25_benchmark/dataset/splitname/train_setlevel.txt' \
                                          --holdout_dataset_path '<your dataset path>/train' \
                                          --holdout_data_list_path '<your project path>/dfc25_benchmark/dataset/splitname/holdout_setlevel.txt' 
```


### `D. Inference & Evaluation`
For current development stage and subsequent test stage, you can run the following code to generate raw & visualized prediction results and evaluate performance
```bash
python script/infer_using_baseline_network.py  --val_dataset_path '<your dataset path>/val' \
                                               --val_data_list_path '<your project path>/dfc25_benchmark/dataset/splitname/val_setlevel.txt' \
                                               --existing_weight_path '<your trained model path>' \
                                               --inferece_saved_path '<your inference results saved path>'
```


### `E. Other Benchmarks & Setup` (🛠️Under Construction)
In addition to the above supervised deep models, BRIGHT also provides standardized evaluation setups for several important learning paradigms and multimodal EO tasks:

* [`Cross-event transfer setup`](): Evaluate model generalization across disaster types and regions. This setup simulates real-world scenarios where no labeled data (**zero-shot**) or limited labeled data (**one-shot**) is available for the target event during training. 

* [`Unsupervised domain adaptation`](): Adapt models trained on source disaster events to unseen target events without any target labels, using UDA techniques under the **zero-shot** cross-event setting.

* [`Semi-supervised learning`](): Leverage a small number of labeled samples and a larger set of unlabeled samples from the target event to improve performance under the **one-shot** cross-event setting.

* [`Unsupervised multimodal change detection`](): Detect disaster-induced building changes without using any labels. This setup supports benchmarking of general-purpose change detection algorithms under realistic large-scale disaster scenarios.

* [`Unsupervised multimodal image matching`](UMIM_benchmark/README.md): Evaluate the performance of matching algorithms in aligning **raw, large-scale** optical and SAR images based on **manual-control-point**-based registration accuracy. This setup focuses on realistic multimodal alignment in disaster-affected areas.



## 🤔Common Issues
Based on peers' questions from [issue section](https://github.com/ChenHongruixuan/BRIGHT/issues), here's a quick navigate list of solutions to some common issues.

| Issue | Solution | 
| :---: | :---: | 
|  Abnormal accuracy (like 0 or -999999) given by leaderboard   |   Keep the prediction name and label name consistent / Zip all prediction files directly, not the folder containing them     |
|  Leaderboard server not responding after submitting results   |   Change browser (Google Chrome recommended)     |
|  Python package conflicts   |   The baseline code is not limited to a specific version, and participants do not need to match the version we provide.     |


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