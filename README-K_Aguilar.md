# Exploring Image Classification with Convoluted Neural Networks


## Objective

For my Module 4 project, I am working with chest x-ray images to perform a binary classification task, determining if the subject of each x-ray has pneumonia. This is a supervised learning exercise and the data was classified into test, train and validation folders by class and downloaded from Kaggle [here.](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Libraries

* For navigating directories: OS, Shutil
* For image display and plotting model performance: Matplotlib, Pillow, Scipy
* For modeling: tensorflow, keras
* For timing models: datetime
* For evaluating models: Scikit-Learn

## Dataset

This dataset consists of 5856 chest x-rays of varying image size. These images were pre-split into test/train/validation folders which I redistributed into a more balanced split of 75/15/15. To maximize the quality of the images, I kept the size of each image at 128 by 128. Additionally, I am using validation and test accuracy as the metric to evaluate the model, though as we are identifying medical conditions recall would also have been acceptable. We are dealing with slight class imbalance, but as it is in favor of our target class I conducted no resampling or downsampling in order achieve more balanced classes. I am also using data augmentation in order to improve the model, as we are only working with a training dataset of ~4000 images, with the aim of avoiding underfitting and ensuring a flexible model.

<img src="images/Sample_XRays.png" alt="Sample X-Rays" style="width: 400px;"/>


## Models

I iterated on a Convoluted Neural Network - both a baseline using my unaugmented data and one with data augmentation, as well as a VGG19 with non-trainable CNN layers. My objective was to build a CNN with 90% accuracy on the test data, as I am using the test and validation datasets to measure the model performance. Although the model is being trained only on the train set to prevent data leakage, I set early stopping parameters and made changes to the hyperparameters based on the model's reduction of loss on the validation data, set with a patience of 20 epochs. W

### Evaluation

My from-scratch CNNs achieved an accuracy of 90% on the validation data most consistently when I had at least 3 convolutional layers and up to 4 dense layers. 

* CNN Model: My baseline model shows evidence of overfitting as shown on the test set of data, although the best saved model achieved 
* CNN Model with Data Augmentation:
* VGG19:


## Further Work
 
Next steps for this model are to continue tuning hyperparameters to improve accuracy and minimize loss on the testing dataset - perhaps adding additional layers to my CNN, freezing layers in the VGG19, and using other pre-trained models such 

## Conclusion