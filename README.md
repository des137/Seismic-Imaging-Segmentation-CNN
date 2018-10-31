# SeismicSaltDetector

## Contents

[**1. Background**](#background)

[**2. Effectiveness of Convolutional Neural Networks**](#cnn)

[**3. U-Net architecture**](#unet)

[**4. Hyperparameter Tuning**](#hyperparameters)

[**5. Metric for Image Segmentation**](#metric)

[**6. Summary**](#results)

[**7. About Me**](#me)


## <a name="background">Background</a>
Reflection seismology is one of the most important techniques to uncover the interiors of the earth for oil and gas explorations. **Seismic images** are obtained by sending artificial seismic waves (earthquakes) inside the earth to accurately probe its interiors. Since, sending mechanical instruments inside the earth is extremely costly and technologically challenging, refection seismology remains the only reliable, cost-effective, and feasible technique for such endeavors. Thus, geophysics industries rely heavily on accurate imaging and analysis of these seismic images to gain information regarding the whereabouts of hydrocarbons (trapped almost hundreds of kilometers deep inside the earth). The raw images which are obtained by combining multiple signals require a lot of preprocessing. This is performed using estimations of sound waves inside the earth using complex geophysical models of the earth. After the preprocessing, these seismic images are in one-to-one correspondence with the true structure of the subsurface. 

An extremely difficult task that follows the preprocessing step is _the interpretation of these seismic images_. Accurate interpretations of the seismic images requires years of experience in the field of geophysics. **The mistakes are costly**. Wrong interpretations can cost millions of dollars to oil and gas companies, if they dig into the earth with false predictions of hydrocarbons. Additionally, there are multiple risks involved while sending the mechanical instruments inside the earth. A very important question in this business, therefore, is whether we can leverage the power of computers to help geophysicists accurately map out the regions in the images that correspond to the presence of hydrocarbons. I take an initial step in this direction by training a neural network on seismic images with training set (of 4000 seismic images) and predicting the outcomes of the network on test set (18000 images) made available via Kaggle. The given seismic images are grayscale images, and each image corresponds to 101 by 101 (total 10201) pixels. 

So in this regard, the **question** that I tackle is: given an image of size 101 by 101 with a grayscale intensity of 1 to 255, can I accurately segment the region of the images that correspond to the presence of hydrocarbons? (Note that the presence of salts inside the earth is an indicator of the presence of hydrocarbons, therefore, I will use the terms 'salts-detection' and 'hydrocarbon-detection', interchangeably). 

## <a name="cnn">Effectiveness of Convolutional Neural Networks</a>
The problem mentioned above, when phrased in different words is: for each of the pixels from the original image, can I classify it as containing salt or not? Such problems are termed 'dense predictions' (since the each pixel of the image needs prediction) or 'image segmentations'. As is well known, the image analysis problems are not properly suited for a regular (dense) neural network, since the input vectors grow exponentially with the size of the image. 'Convolutional neural networks' have proved immensely effective in image, speech, natural language processing tasks. These networks mimic the architecture of the visual cortex of the brain that is responsible for the image recognition. 

## <a name="unet">U-Net architecture</a>

## <a name="hyperparameters">Hyperparameter Tuning</a>

## <a name="metric">Metric for Image Segmentation</a>

## <a name="results">Summary and Future Directions</a>
Images in the first column are actual seismic images. The second and third columns show the corresponding labeled masks and the masks predicted by CNNs, respectively.
<p align="center">
  <img src="https://github.com/des137/SeismicSaltDetector/blob/master/real-masks-predicts.png" width="350">
</p>

## How to run this code on your personal computer:
1. _SeismicSaltDetector_ can be run by simply typing the following commands in terminal: 
```
git clone https://github.com/des137/SeismicSaltDetector.git
cd SeismicSaltDetector
./run.sh
```

## <a name="me">About Me</a>
My name is Amol Deshmukh. I am a physicist and a data scientist. During my PhD, I investigated magnetic properties of neutrons and protons (and other less-known particles called 'hyperons'). I am extremely passionate about applying my skills that physics has taught me over the years to tackle challenging business-oriented problems in data science and machine learning.

## 
