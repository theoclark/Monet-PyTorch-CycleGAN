import torch
from torch import nn
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from generator import Generator
class Model():

  def __init__(self, weights_path, input_image_path, output_image_path):
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      self.input_image_path = input_image_path
      self.output_path = output_image_path
      self.model = Generator().to(self.device)
      self.load_weights(weights_path)

  def load_weights(self, weights_path):
      self.model = torch.load(weights_path, map_location=torch.device(self.device))

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
