# TCNN
# This reposity contains the source code to replicate the numerical tests in "Multiple-Scattering Media Imaging via End-to-End Neural Network"
TCNN.tensorflow
![](https://github.com/Ziyang1992/TCNN/blob/master/1.png)
Tensorflow implementation of TCNN Networks for Multiple Scattering imaging. 

1. Prerequisites
2. Results

Prerequisites

- The results were obtained after training for ~6-7 hrs on a 12GB NVIDIA 1080Ti.
- The code was originally written and tested with tensorflow 1.6.0 and python3.6.4. 
- To train model simply execute python multiple scattering imaging.py
- The dataset can be download in https://rice.app.box.com/v/TransmissionMatrices

Results

Results were obtained by training the model in different imagesize 16x16, 40x40. The size of speckle pattern for each image is 256x256. 
![](https://github.com/Ziyang1992/TCNN/blob/master/5.png)
![](https://github.com/Ziyang1992/TCNN/blob/master/4.png)
![](https://github.com/Ziyang1992/TCNN/blob/master/3.png)


 

      


