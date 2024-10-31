# Facial Expression Recognition 

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
## Installation

This project relies on key machine-learning libraries for model training, evaluation, and data handling. Below are the main libraries used and their versions:

- **PyTorch (1.8.1)**: Used for model building, training, optimization, and leveraging GPU for parallel processing
- **Torchvision (0.9.1)**: Supports dataset loading and data preprocessing (transforms)
- **Scikit-learn (1.0.2)**: Provides metrics like the confusion matrix and F1 score for model evaluation
- **Matplotlib (3.6.0)**: Visualizes model performance metrics during training

Before running the project, ensure you have Python installed on your machine. You can install the required packages by running:


    pip install -r requirements.txt


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

Please refer to our report for more detailed information about the datasets and our approach.

## Model Architecture

<img src= https://github.com/user-attachments/assets/588c15da-bb25-4818-8de1-3c91d7f98585 width="800"/>

The POSTER model utilizes a two-stream architecture combining facial landmark and image feature information using cross-fusion transformer blocks. 

These transformer blocks facilitate the integration of geometric and textural data, addressing common challenges in FER, such as:

- Inter-class similarity where visually similar expressions (e.g., anger vs. sadness) may confuse the model.
- Intra-class variation due to individual differences in facial structure, age, or ethnicity.
- Scale sensitivity, crucial for robust recognition across varying image resolutions.

### Technical Specifications:
- Pyramid Structure: Allows the model to process features at multiple scales, providing a fine balance between context and detail.

- Training and Inference Time: Each epoch requires approximately 4 hours on an NVIDIA Tesla V100 GPU with a batch size of 32, with an average inference time of 2 seconds per image.


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
