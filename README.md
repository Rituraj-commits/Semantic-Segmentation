# Semantic Segmentation on GTAV-to-Cityscapes Labels

The GTAV dataset consists of 24966 synthetic images with dense pixel level semantic annotations. The images have been rendered using the open-world video game Grand Theft Auto 5 and are all from the car perspective in the streets of American-style virtual cities. There are 35 semantic classes which are compatible with the ones of CamVid and Cityscapes dataset. We use Unet++ as our baseline in this scenario. UNet++ is an extension over UNet which uses dense skip connections to extract better feature information. 

## Getting Started

### Dependencies

* PyTorch 1.8.0
* CUDA 10.2
* TensorBoard 2.5.0

### Installing

```
pip install segmentation_models_pytorch
```

### Execution


```
 python train.py
```
```train.py``` contains code for training the model and saving the weights.

```loader.py``` contains code for dataloading and train-test split.

```utils.py``` contains utility functions.


#### Training
We use Adam optimizer for optimizing the objective function. The learning rate is initially set to 0.001 and is halved if the validation loss plateaus for 10 epochs. We train the network until 100 epochs and the best weights are saved accordingly. We use NVIDIA Tesla P100 with 16 GB of VRAM to train the model.

### Quantative Results
We evaluate the model on the basis of Dice Score Coefficient(DSC) and Intersection over Union(IoU) over 35 classes.
<div align="center">
  <img src="https://github.com/Rituraj-commits/Semantic-Segmentation/blob/main/resources/table.png">
  <br>
  <br>
  <em align="center"></em>
  <br>
</div>

### Qualitative Results
<div align="center">
  <img src="https://github.com/Rituraj-commits/Semantic-Segmentation/blob/main/resources/Results.png">
  <br>
  <br>
  <em align="center">Fig 1: Semantic Segmentation on GTAV dataset </em>
  <br>
</div>

## Acknowledgments

[1] [Aerial Segmentation using UNet](https://www.kaggle.com/ayushdabra/inceptionresnetv2-unet-81-dice-coeff-86-acc)

[2] [UNet Multi Class PyTorch](https://github.com/France1/unet-multiclass-pytorch)
