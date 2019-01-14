# VGG16
from keras.applications.vgg16 import VGG16

vgg16_model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

# VGG19
from keras.applications.vgg19 import VGG19

vgg19_model = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

import torchvision.models as models

model = models.vgg16(pretrained=True)
