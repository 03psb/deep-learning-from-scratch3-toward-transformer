import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero.datasets import get_mnist
from dezero.dataset import DatasetLoader

batch_size = 10
max_epoch = 1

train, test = get_mnist()
train_loader = DatasetLoader(train, batch_size)
test_loader = DatasetLoader(test, batch_size, shuffle=False)

for epoch in range(max_epoch):
    for x, t in train_loader:
        print(x.shape, t.shape)
        break

    for x, t in test_loader:
        print(x.shape, t.shape)
        break