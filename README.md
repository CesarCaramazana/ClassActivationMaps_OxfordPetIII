# Class Activation Maps in Oxford Pet III Dataset

This notebook is my personal experimentation with Class Activation Maps (https://arxiv.org/abs/1512.04150), using the Oxford III Pet dataset (available here: https://www.kaggle.com/tanlikesmath/the-oxfordiiit-pet-dataset) for the evaluation.

**Disclaimer: this notebook is yet to be finished. So far, the model, the get_cam function and the training loop have been defined.*.

Class Activation Maps are a way of understanding what a Convolutional Neural Network "sees" in order to perform image classification.  They are very useful to analyze where the attention is placed and which are the key regions that help the network discriminate among classes. 


The ultimate milestone of this project is to propose a weakly supervised segmentation algorithm that performs region growing on the Class Activation Maps, using a hybrid arquitecture that performs both image classification and semantic segmentation. If the experiments are succesful in the Oxford Pet dataset, we will switch to ISIC DB 2018, a dermoscopic imaging dataset for early detection of melanoma (available here: https://challenge.isic-archive.com/data)
