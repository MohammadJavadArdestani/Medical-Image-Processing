# Medical Image Processing

In this Image processing project, I used a CNN classifier to classify **COVID-19** Infected Lung Xray images from **Healthy** Lung Xray images.

## Table of Contents
* [DataSets](https://github.com/MohammadJavadArdestani/Medical-Image-Processing-/edit/main/README.md#datasets)
* [Deep Learning Model](https://github.com/MohammadJavadArdestani/Medical-Image-Processing-/edit/main/README.md#deep-learning-model)
* [Evaluation](https://github.com/MohammadJavadArdestani/Medical-Image-Processing-/edit/main/README.md#evaluation)
<!-- * [Appendix](https://github.com/MohammadJavadArdestani/Medical-Image-Processing-/edit/main/README.md#appendix) -->

## DataSets
I selected my dataset from three different sources: 
- Cohen's [COVID Chest X-ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset) 
- Paul Mooney's [Chest X-ray Dataset (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [COVID-19_Radiography_Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
<br>

Train Dataset contains 850 images by the following distribution: <br> <br>
<tab><tab>![train_dataset](https://github.com/MohammadJavadArdestani/Medical-Image-Processing-/blob/main/Pictures/train_dataset.png)


Some samples of both groups: <br>
<tab><tab>![Positive_samples](https://github.com/MohammadJavadArdestani/Medical-Image-Processing-/blob/main/Pictures/covid_positive.png)<br><br>
<tab><tab>![Positive_samples](https://github.com/MohammadJavadArdestani/Medical-Image-Processing-/blob/main/Pictures/covid_negative.png)

## Deep Learning Model


* Pre-trained DenseNet-121 is used as the core here for our Deep Learning Model (More details [here](https://arxiv.org/abs/1608.06993)).
* I used pre-trained weights as a means to Transfer Learning. To learn and achieve higher accuracy on our model faster.

* Instead of freezing CNN Layers and training only the Fully Connected Layer (like most people do in Classification Task), I trained all the layers,  including CNNs and Classification layer.

![DenseNet-121](https://miro.medium.com/max/1400/1*vIZhPImFr9Gjpx6ZB7IOJg.png)


## Evaluation 

Test Dataset contains 200 images from each group. 

Classification results:
```
              precision    recall  f1-score   support

           0       0.73      0.95      0.83       200
           1       0.94      0.66      0.77       200
    accuracy                           0.81       400
   macro avg       0.84      0.80      0.80       400
weighted avg       0.84      0.81      0.80       400
```
<br><br>
Confusion Matrix: <br><br>
<tab><tab>![Positive_samples](https://github.com/MohammadJavadArdestani/Medical-Image-Processing-/blob/main/Pictures/Confusion_Matrix.png)

  ## Appendix
  Many thanks to [Arun Pandian R](https://www.kaggle.com/arunrk7) for his useful tutorial.
  
