import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.dataset import DatasetLoader
from dezero.models import MLP


max_epoch = 3
batch_size = 100

train_set, test_set = dezero.datasets.get_mnist()
train_loader = DatasetLoader(train_set, batch_size)
model = MLP((784, 1000, 10))
optimizer = optimizers.SGD().setup(model)

# model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    print('epoch: {}, loss: {:.4f}'.format(
        epoch + 1, sum_loss / len(train_set)))

model.save_weights('my_mlp.npz')