# Medical Image Processing

In this Image processing project, I use a CNN classifier to classify **COVID-19** Infected Lung Xray images from **Healthy** Lung Xray images.

## Table of Contents
* []()
* []()
* []()
* []()

## DataSets
I selected my datset from three different sources: 
- Cohen's [COVID Chest X-ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset) 
- Paul Mooney's [Chest X-ray Dataset (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [COVID-19_Radiography_Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
<br>

Train Dataset contains 850 images by following distribution: 
<link>


Some samples of both groups: 

<link>

## Models

* pre-trained DenseNet-121 is used as the core here for our Deep Learning Model (More details [here](https://arxiv.org/abs/1608.06993)).

* Instead of freezing CNN Layers and training only the Fully Connected Layer we traning all the the Layers,  including CNNs and Classification layer.

![DenseNet-121](https://miro.medium.com/max/1400/1*vIZhPImFr9Gjpx6ZB7IOJg.png)


## Evaluation 

Test Dataset contains 200 images from both groups. 

Classification results:
```
              precision    recall  f1-score   support

           0       0.73      0.95      0.83       200
           1       0.94      0.66      0.77       200
    accuracy                           0.81       400
   macro avg       0.84      0.80      0.80       400
weighted avg       0.84      0.81      0.80       400
```

Confusion Matrix: 

<link>