# Monet-PyTorch-CycleGAN
PyTorch implementation of [CycleGAN](https://arxiv.org/abs/1703.10593) to convert images to Monet-style paintings. Read more about the project on my [blog](https://theoclark.co.uk/posts/cycle-gan.html).

## Samples

![samples](https://github.com/theoclark/Monet-PyTorch-CycleGAN/blob/main/SampleImages.png)

## Usage

### Inference

To see the model in action and convert images, head over to this [site](http://monet.theoclark.co.uk/) (hosted on heroku and built using Flask).

### Training

Use the train.ipynb notebook for training (located in the 'Model' directory). To download the data you will need a [Kaggle](https://www.kaggle.com/) account and API key (accessible through account settings). The notebook is set up to run through Google Colab and the api key (kaggle.json) needs to be stored in the root file of your Google Drive.
