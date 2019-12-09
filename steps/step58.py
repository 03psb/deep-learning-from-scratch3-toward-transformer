import numpy as np
from PIL import Image
import dezero
from dezero.models import VGG16
from dezero.dataset import preprocess_vgg
from dezero.datasets import get_imagenet_labels


url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
x = preprocess_vgg(img)
x = x[np.newaxis]

labels = get_imagenet_labels()

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)

predict_id = np.argmax(y.data)
print(labels[predict_id])
img.show()