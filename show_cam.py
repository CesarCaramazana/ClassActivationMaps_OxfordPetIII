import torch
import torchvision
import torch.nn as nn

def show_cam(cam):
  """
    Visualize the first group of Class Activation Maps in a batch. Plots the class activation maps for the 2 classes: cat, dog.
      input: cam #(batch, 6, h, w)
  """
  batch, classes, h, w = cam.shape
  cam = torch.from_numpy(cam) #to Tensor

  upsample = nn.UpsamplingBilinear2d(scale_factor=16) #Rescale CAMs to the input/output resolution.
  cam = upsample(cam)


  plt.figure(figsize=(8,4))
  plt.subplot(121)
  plt.title("Cat")
  plt.imshow(cam[0, 0], cmap='gnuplot2')
  plt.subplot(122)
  plt.title("Dog")
  plt.imshow(cam[0, 1], cmap='gnuplot2')
  plt.show()
