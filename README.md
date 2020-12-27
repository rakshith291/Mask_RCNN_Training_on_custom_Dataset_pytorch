# Mask_RCNN_Training_on_custom_Dataset_pytorch


This repo is based on Pytorch tutorial for Image/video detection https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html  

In this repo I tried to fine-tune Mask RCNN for custom based dataset.

Here It's important to understand how to feed the data to the model  
I have inherited the proprties of  torch.utils.data.Dataset class   
we need to overwrite __init__, __call__,__getlen__ function.  
In __call__ if index of an image is given it should be able to read the image and  
corresponding target and pass the image to transforms/augumentation before return.  

Then dataloader will take care of the input pipeline , afterwards https://github.com/pytorch/vision/tree/master/references/detection   
use the above link to get scripts which gives utility function for object detection.

Coco pre-trained weight can be obtained from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html  
coco can be installed by using https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI  
install cython if make is failing 
