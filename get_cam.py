import torch
import torchvision
import torch.nn as nn

def get_cam(model, feature_map, num_classes=2):
  """
    inputs:
      · model: net() object with the parameters for all layers. Used to recover weights from GAP --> FC layer
      · feature_map: last convolutional volume of encoder, with shape [batch, channels, h,w], where channels is
        a design parameter.
        - h,w ~ (16,16) or (8,8), output stride=16 (wrt input size)
      · num_classes: number of classes. *Update needed to be class agnostic. Which variable here contains the num of classes?
    output: cam #[batch, num_classes, h, w]. Class activation maps for each class and batch
  """
  params = list(model.parameters())
  w = np.squeeze(params[-2].data.cpu().numpy()) #(6, 1536) e.g
  batch, channels, height, width = feature_map.shape

  cam = np.zeros((batch, num_classes, height, width))

  for batch in range(batch):
    f_b = feature_map[batch].detach().cpu().numpy()
    f_b = f_b.reshape(channels, height*width)

    cam_i = np.dot(w, f_b) #(2, 256)
    cam_i = cam_i.reshape(num_classes, height, width) #(2, 16, 16)

    cam[batch] = cam_i #(batch, 2, 16, 16)

  return cam
