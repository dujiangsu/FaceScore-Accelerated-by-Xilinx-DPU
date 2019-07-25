# FaceScore-Accelerated-by-Xilinx-DPU

## Project Describtion

This project is a casestudy completed in Xilinx Internation Summer School.

![Overall Architecture](https://github.com/dujiangsu/FaceScore-Accelerated-by-Xilinx-DPU/blob/master/images/Arch.png)


As shown in Figure 1, ARM(PS) is used to process the video flow and convert it to images. Then these images will be loaded into next module - Face Detection, which is actually a deep learning application. For this module, its computation-intensive part (convolutional operation) will be loaded into DPU (a soft core implemented in FPGA, programmable logic). 

The face detection module produces the coordinates of faces in an image, using which ARM will abstract face images. Having face images, we can make scores. In other words, these faces will be put into next module - Face Score, which is substantially a image classification CNN. After scoring, we convert these images to video.

## Implementation
The face detection module is githubed from [Xilinx repository](https://github.com/Xilinx/Edge-AI-Platform-Tutorials/tree/master/docs/DPU-Integration). 

![Image preprocessing](https://github.com/dujiangsu/FaceScore-Accelerated-by-Xilinx-DPU/blob/master/images/image%20preprocessing.png)

The face score module is trained and converted to DPU-usable version by ourselves. We introduce our optimization steps below:
 - First, we tried Alexnet, however the weight amount and computing amount is too heavy for Ultra96 to stand. So we find a small CNN called miniVggNet. 
 - Second, the original dataset we collecte from github is 192*192, and it will largely increase the size of network. In this way, we resize the image to 32*32, as shown in Figure 3/4.
 - Third, we use tools provided by xilinx DNNDK to quantify weight in the trained model and further decrease the final ELF file size. 
 
 ## Directory Introduction
 The *face_dection* directory is the complete and runnable function of face dection.
 The *miniVGGtoELF* directory complete the whole process from Tensorflow Training, DNNDK quantilization, fine-tuning, and convert to ELF. 
 The *face_score* directory is the runnable function of face score, unfortunately, because DPU only process the convolutional part of a CNN , the final output the last conv layer and we need to softmax it to the final classfication by hand. Due to the unclear doc provided, we didn't complete it. 
 
 ## Conclusion 
 
 Xilinx Summer School provides me a good experience on the end-to-end deployment of FPGA && Deep Learning and it is worth experiencing. The tool chain of deploying a CNN on FPGA is becoming ever mature. However, from my own perspective, the price, develop efficiency, and flexibility can still block the way for FPGA to general computing.
 
 This is a very brief introduction of our project, welcome to discuss for more info.
