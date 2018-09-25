# TCNN
# This reposity contains the source code to replicate the numerical tests in "Multiple-Scattering Media Imaging via End-to-End Neural Network"
TCNN.tensorflow

Tensorflow implementation of TCNN Networks for Multiple Scattering imaging. 

1. Prerequisites
2. Results
3. Observations
4. Useful links

Prerequisites

- The results were obtained after training for ~6-7 hrs on a 12GB NVIDIA 1080Ti.
- The code was originally written and tested with tensorflow 1.6.0 and python3.6.4. 
- To train model simply execute python multiple scattering imaging.py
- The dataset can be download in https://rice.app.box.com/v/TransmissionMatrices

Results

Results were obtained by training the model in batches of 2 with resized image of 256x256. Note that although the training is done at this image size - Nothing prevents the model from working on arbitrary sized images. No post processing was done on the predicted images. Training was done for 9 epochs - The shorter training time explains why certain concepts seem semantically understood by the model while others were not. Results below are from randomly chosen images from validation dataset.

Pretty much used the same network design as in the reference model implementation of the paper in caffe. The weights for the new layers added were initialized with small values, and the learning was done using Adam Optimizer (Learning rate = 1e-4). 

   

   

   

   

   

Observations

- The small batch size was necessary to fit the training model in memory but explains the slow learning
- Concepts that had many examples seem to be correctly identified and segmented - in the example above you can see that cars, persons were identified better. I believe this can be solved by training for longer epochs.
- Also the resizing of images cause loss of information - you can notice this in the fact smaller objects are segmented with less accuracy.



Now for the gradients,

- If you closely watch the gradients you will notice the inital training is almost entirely on the new layers added - it is only after these layers are reasonably trained do we see the VGG layers get some gradient flow. This is understandable as changes the new layers affect the loss objective much more in the beginning.
- The earlier layers of the netowrk are initialized with VGG weights and so conceptually would require less tuning unless the train data is extremely varied - which in this case is not.
- The first layer of convolutional model captures low level information and since this entrirely dataset dependent you notice the gradients adjusting the first layer weights to accustom the model to the dataset.
- The other conv layers from VGG have very small gradients flowing as the concepts captured here are good enough for our end objective - Segmentation. 
- This is the core reason Transfer Learning works so well. Just thought of pointing this out while here.

      


