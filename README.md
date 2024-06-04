# Deep-Learning-Reproducibility-Project

This project aims to reproduce the key findings and claims of the paper "POSTER: A Pyramid Cross-Fusion 8 Transformer Network for Facial Expression Recognition." 

```
Deep-Learning-Reproducibility-Project/
├── checkpoint/
├── data/
│   ├── Aihub train
│   ├── AIhub test
├── data_preprocessing/
├── log/             
│   ├── Training logs, graphs
├── models/          
│   ├── Model codes used for training
├── main.py
├── main_no_augmentation.py
└── requirements.txt
```

## Train

    python main.py --data path/to/dataset --lr 3.5e-5 --batch-size 64 --epochs 100 --gpu 0

## Test

    python main.py --data path/to/dataset --evaluate path/to/checkpoint

## Datasets 

The original paper provided a dataset for their research, but it was inaccessible to us. As a result, we constructed our own datasets for our experiments: 

1. AI Hub - Composite Images for Korean Emotion Recognition
 
   This dataset can be accessed from the following link: [AI Hub Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=82)

2. Custom Dataset

    We collected additional data directly by capturing images of VR users wearing headsets. Due to privacy and consent considerations, we are not publicly sharing this dataset. However, if you need to access it, please contact us via email. (daun@seoultech.ac.kr)

For both datasets, we focused on the lower part of the face images, as our goal was to recognize emotions of VR users based on the visible facial area when wearing a VR headset. Please refer to our report for more detailed information about the datasets and our approach.

## Checkpoints

The checkpoints we trained are available at the following:
[Checkpoints](https://drive.google.com/drive/folders/1s55acYF6KqU9yJ-z909Oe1CF-kQtWFul?usp=drive_link)

| Train Dataset                  | Top-1 Accuracy | Path                                 | 
| --------------------------     | -------------- | ------------------------------------ |
| AI Hub                         | 100            | checkpoint/try1_aihub_model_best.pth |
| Ours                           | 81.40          | checkpoint/try3_ours_best.pth        |
| Ours(data augmentation)        | 92.81          | checkpoint/try4_ours_best.pth        |
| Ours(parameter_search)         | 80.77          | checkpoint/try5_ours_best.pth        |



## Reference
[POSTER_V2](https://github.com/Talented-Q/POSTER_V2)

@article{mao2023poster,
  title={POSTER V2: A simpler and stronger facial expression recognition network},
  author={Mao, Jiawei and Xu, Rui and Yin, Xuesong and Chang, Yuanqi and Nie, Binling and Huang, Aibin},
  journal={arXiv preprint arXiv:2301.12149},
  year={2023}
}
