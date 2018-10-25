# SeismicSaltDetector

## Contents

[**1. Background**](#background)

[**2. Challenges in Seismc Imaging**](#seism)

[**3. Effectiveness of Convolutional Neural Networks**](#cnn)

[**3. U-Net architecture**](#unet)

[**4. Hyperparameter Tuning**](#hyperparameters)

[**5. Metric for Image Segmentation**](#metric)

[**6. Summary and Future Directions**](#results)

[**7. About Me**](#kaka)


## <a name="background">Background</a>
Seismic images are obtained by sending artificial seismic waves inside the earth to accurately probe its interiors, for oil and gas exploration purposes. 

## <a name="seism">Challenges in Seismc Imaging</a>
Correct segmentation of the salt (that traps oil and gas) in these images is a major challenge. Analysis of these seismic images usually requires years of experience in the field of geophysics. The natural question therefore is, whether machines can learn the pattern in the seismic images and perform the required segmentation. 

## <a name="cnn">Effectiveness of Convolutional Neural Networks</a>
We take a step toward achieving this by using a convolutional neural network approach. 

## <a name="unet">U-Net architecture</a>

## <a name="hyperparameters">Hyperparameter Tuning</a>

## <a name="metric">Metric for Image Segmentation</a>

## <a name="results">Summary and Future Directions</a>
Images in the first column are actual seismic images. The second and third columns show the corresponding labeled masks and the masks predicted by CNNs, respectively.
<p align="center">
  <img src="https://github.com/des137/SeismicSaltDetector/blob/master/real-masks-predicts.png" width="350">
</p>

## Google Colab commands
1. To run on Google Colab, type the following commands: 
```
!git clone https://github.com/des137/SeismicSaltDetector.git
import os
os.chdir('SeismicSaltDetector')
!./run.sh
```

## <a name="kaka">About Me</a>
My name is Amol Deshmukh. I am a physicist and a data scientist. During my PhD, I investigated magnetic properties of neutrons and protons (and other less-known particles called 'hyperons'). 
