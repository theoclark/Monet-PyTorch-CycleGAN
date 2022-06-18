import torch
from torch import nn
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

class Downsample_block(nn.Module):
  def __init__(self, n_channels):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(n_channels, 2*n_channels, 3, stride=2, padding=1),
        nn.InstanceNorm2d(n_channels),
        nn.ReLU()
    )

  def forward(self, X):
    return self.model(X)

class Upsample_block(nn.Module):
  def __init__(self, n_channels):
    super().__init__()
    self.model = nn.Sequential(
        nn.ConvTranspose2d(2*n_channels, n_channels, 3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(n_channels),
        nn.ReLU()
    )

  def forward(self, X):
    return self.model(X)

class Residual_block(nn.Module):
  def __init__(self, n_channels):
    super().__init__()
    self.model = nn.Sequential(
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=0),
        nn.InstanceNorm2d(n_channels),
        nn.ReLU(),
        nn.ReflectionPad2d(padding=1),
        nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=0),
        nn.InstanceNorm2d(n_channels),
    )

  def forward(self, X):
    return self.model(X) + X

class Generator(nn.Module):
    """
    As described in the CycleGAN paper: https://arxiv.org/abs/1703.10593
    Based on the following paper: https://arxiv.org/abs/1603.08155
    """
    def __init__(self):
      super().__init__()
      self.model = nn.Sequential(
          nn.ReflectionPad2d(padding=3),
          nn.Conv2d(3, 64, 7, stride=1, padding=0),
          nn.InstanceNorm2d(64),
          nn.ReLU(),
          Downsample_block(64),
          Downsample_block(128),
          Residual_block(256),
          Residual_block(256),
          Residual_block(256),
          Residual_block(256),
          Residual_block(256),
          Residual_block(256),
          Residual_block(256),
          Residual_block(256),
          Residual_block(256),
          Upsample_block(128),
          Upsample_block(64),
          nn.ReflectionPad2d(padding=3),
          nn.Conv2d(64, 3, 7, stride=1, padding=0),
          nn.Sigmoid(),
      )

    def forward(self, X):
      return self.model(X)

class Model():


  def __init__(self, weights_path, input_image_path, output_image_path):
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      self.input_image_path = input_image_path
      self.output_path = output_image_path
      self.model = Generator().to(self.device)
      self.load_weights(weights_path)

  def load_weights(self, weights_path):
      self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))
      # self.model = torch.load(weights_path, map_location=torch.device(self.device))

  def predict(self, image_path):
      transform_image = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(256),
          transforms.ToTensor()
          ])
      input_image = Image.open(image_path)
      input_image = transform_image(input_image)
      save_image(input_image, self.input_image_path)
      output_image = self.model(input_image.unsqueeze(0).to(self.device)).cpu().squeeze().detach()
      save_image(output_image, self.output_path)
