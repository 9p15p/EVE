# `🍎`EVE: Everything in Video can be Segmented End-to-End.

**🍎EVE**(**E**verything in **V**ideo can be Segmented **E**nd-to-End) is a simple toy aimed at automatically or
interactively
segmenting anything in videos. It incorporates algorithms
including [SAM(Segment Anything)](https://github.com/facebookresearch/segment-anything)
and [XMem(Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model)](https://github.com/hkchengrex/XMem).
The native SAM is used to encode and decode image information and produce single frame predictions, while XMem equips
EVE with the ability to integrate temporal information. 🍎EVE can be trained end-to-end, allowing users to fine-tune it
on their own datasets easily.

## 🚀Updates

* [2023/05/18] We release 🍎EVE!

## 😄requirements

```shell
conda create -n EVE python=3.8 -y
conda activate EVE
# XMem
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv-python
pip install -r requirements.txt

# SAM
# SAM we use is modified, 
# do not run 'pip install git+https://github.com/facebookresearch/segment-anything.git'
pip install pycocotools matplotlib onnxruntime onnx

# davis2017evaluation
cd davis2017evaluation
python setup.py install

```

## 🎏Getting Started

### Video Tutorials

1. Segment everything in video.

   ![EVE_demo_0.mp4](docs/docs_EVE/EVE_demo_0.mp4)
2. Segment specific object in video.

   ![EVE_demo_1.mp4](docs/docs_EVE/EVE_demo_1.mp4)

### Detailed Tutorails

Please refer to [TOTARIALS.md](docs/TUTORIALS.md) for more details

## ⛵️Experiments on VOS

### Datasets

If you want to retrain 🍎EVE on VOS-Datasets(e.g. DAVIS2017 or YouTubeVOS2019), you need to structure datasets(DV17,
YV18, YV19) as follows.

```
DATASETS
├── DAVIS2017
│   ├── Annotations
│   │   └── 480p [150 entries]
│   ├── ImageSets
│   │   ├── 2016
│   │   │   ├── train.txt
│   │   │   └── val.txt
│   │   └── 2017
│   │       ├── test-challenge.txt
│   │       ├── test-dev.txt
│   │       ├── train.txt
│   │       └── val.txt
│   └── JPEGImages
|       └── 480p [150 entries]
└── YoutubeVOS2019
    ├── test
    │   ├── Annotations [541 entries]
    │   ├── JPEGImages [541 entries]
    │   └── meta.json
    ├── train
    │   ├── Annotations [3471 entries]
    │   ├── JPEGImages [3471 entries]
    │   └── meta.json
    └── valid
        ├── Annotations [507 entries]
        ├── JPEGImages [507 entries]
        └── meta.json


```

### Retrain on VOS

Please refer to [train_s2.sh](sam_scripts/train_s2.sh) for more details.

### Inference on VOS

```

python sam_scripts/eval_EVE.py \
--eval_on_dv17 \
--model_type vit_h \
--output output/D17_val_EVE \
--model saves/EVE.pth

```

Performances on DAVIS2017-val


| J&F-Mean |   J-Mean | J-Recall | J-Decay | F-Mean | F-Recall | F-Decay |
| :------- | -------: | :------: | :------: | :-----: | :------: | :------: |
| 0.831953 | 0.805476 | 0.893966 | 0.071745 | 0.85843 | 0.929103 | 0.098502 |

## 🔧TODO List

* [ ]  enable 🍎EVE to delete object.
* [ ]  develop a function to utilize interactive masks or strokes to guide 🍎EVE.
* [ ]  enbale 🍎EVE to refine the masks in the intermediate process of inference.
* [ ]  create a local interactive_demo.
* [ ]  train 🍎EVE by the data engine proposed by [SAM](https://github.com/facebookresearch/segment-anything).

## 👏 Acknowledgements

This project is based on [XMem](https://github.com/hkchengrex/XMem)
and [Segment-Anything](https://github.com/facebookresearch/segment-anything). Thanks for their outstanding work.
