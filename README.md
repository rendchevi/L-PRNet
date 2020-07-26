# L-PRNet

## Project Description
L-PRNet (Lightweight Position Regression Network) is a lightweight adaptation and re-implementation of PRNet, a deep learning model designed to perform 3D face reconstruction given only a single RGB image of the face.  
  
You can check the original PRNet author's repo [here](https://github.com/YadiraF/PRNet) and their very well written paper [here](https://arxiv.org/abs/1803.07835).  
  
I was so impressed by the author's works while back when it first came out, their solution for 3D face reconstruction is so straight-forward and elegant by creating a novel 
2D represention of the 3D face structure called UV position map of which one can directly regress 2D facial image into it's 3D structure with low computational cost.  

This project is concerned on making 3D dense face reconstruction lightly run on CPU device, specifically on the web and mobile devices.  
It'd probably a bit of a ramble to specify what I actually want to achieve with L-PRNet but in short, I intend to make 3D face reconstruction accessible on mobile devices for AR/VR/MR and creative purposes.  

To do that, I think improving PRNet efficiency by reducing complexities of the input-output data and the network architecture might be the answer.  
  
In terms of speed, the original PRNet is already awesome capable to achieve real-time inference on GPU device (100 fps with GTX 1080).  
But, PRNet inference on CPU device isn't fast enough nor achieve real-time which would be problematic if we want to run it on the web and mobile devices.  
  
In this repo, I provide my early iteration of the L-PRNet (pre-trained mode and some testing codes).  
Feel free to experiment it with yourself or reach me if you've interest to improve (or currently on) the project.

### How to use the model
Below are the following dependencies needed:
```
tensorflow=2.2.0 # (tensorflow-cpu is also fine)
moviepy # to test on video
```
Python file ```inference.py``` contain LPRNet class which you can utilize for the following:
```
```
I provided a testing script to utilize the model to predict 3D face in a video (you need ```moviepy``` module for this):
```
```

## Project Details

### Difference between L-PRNet and PRNet architecture
- L-PRNet's input and output dimension is ```128x128x3```
- L-PRNet's encoder uses a MobileNet-like architecture, utilizing a separable convolution to make the network more efficient.
- L-PRNet's decoder uses a simpler (shallower) stack of transposed convolutions.

Rough comparison of both model in terms of inference speed:
|                 | L-PRNet |  PRNet | 
| -------         | ------- | ------ |
| CPU             |         |        |
| GPU             |         |        |
| # of Parameters |         |        |

### Dataset Preparation and Preprocessing
Data preparation and preprocessing process are similar with the original PRNet. Dataset used is 300WLP dataset, specifically I used the whole 300WLP subset which are HELEN, IBUG, AFLW, and LFWP dataset. The dataset contains around 100k face images in various poses along with their 3DMM (3D Morphable Model) parameters which would be used to reconstruct the ground truth 3D structure/model of the face.  
  
For the network's input, 2D face images from the dataset are directly used (images are normalized ```[0, 1]```). The network's output is a 2D representation of 3D structure of the face, called the UV position map. Basically, UV position map is a 2D projection of the 3D face structure onto the UV space. To generate the UV position map, I used the author's python modules to process 3DMM into UV position map.  
  
UV position map is a ```128x128x3``` array where each channel represents 3D (x,y,z) coordinates, we can simply reconstruct the 3D structure from the UV map (in the form of point cloud) via reshaping it to ```128*128x3``` array.  
  
Below is a sample of the input-output pair.  
  
In the original PRNet paper, the author perform data augmentation process (color scaling, translation, and rotation) which I haven't done it yet. I plan to re-train the model soon with data augmentation and update the pre-trained model in this repo.

### Network Architecture and Training
The architecture of L-PRNet is similar to PRNet, a CNN autoencoder.  
The encoder part is composed of a stack of 2D separable convolutions, similar to MobileNet architecture.  

The decoder part is composed of a stack of 2D transposed convolutions, which I made shallower compared to the original PRNet to reduce computational cost.
  
Adam is used as the optimizer, the learning rate starts at ```0.0001``` and decay half after 5 epochs as done in the original PRNet, batch size used is 16. 
The pre-trained model in this repo have been trained for 15 epochs.

### Future Improvements
- [ ] Re-train the model with augmented data suggested in the original paper for handling more difficult situation
- [ ] Evaluate the model with NME as done in the original PRNet paper
- [ ] Create applet for an interactive 3D reconstruction
- [ ] (This one's a bit of a longshot) Study on applying similar 2D UV representation of other human body parts like full head, body, or hands.
