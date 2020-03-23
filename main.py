import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from utils import RunBuilder
from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(linewidth=120)
print('torch.version:', torch.__version__)
print('torchvision.version', torchvision.__version__, '\n')

# 1. ============== ETL =======================
print('1. ============== ETL =======================')
# downloading FashionMNIST if not already downloaded and putting it into data folder (E and T)
print('Getting the training data..')
train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

print('Getting the test data..')
test_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# 2. ============== Build the model =======================
print('2. ============== Building the model =======================')


class CNN(nn.Module):
    """
    Class definition of the model we will be using in the training.
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, tens):
        """
        input tensor forward propagation implementation.
        """
        # hidden conv1 layer
        tens = self.conv1(tens)
        tens = F.relu(tens)
        tens = F.max_pool2d(tens, kernel_size=2, stride=2)

        # hidden conv2 layer
        tens = self.conv2(tens)
        tens = F.relu(tens)
        tens = F.max_pool2d(tens, kernel_size=2, stride=2)

        # hidden fc1 layer
        # flatten the output from previous conv layers
        tens = tens.reshape(-1, 12 * 4 * 4)
        tens = self.fc1(tens)
        tens = F.relu(tens)

        # hidden fc2 layer
        tens = self.fc2(tens)
        tens = F.relu(tens)

        # output layer
        # NB: output layer does not get activated because it will be later in the loss computation
        tens = self.out(tens)

        return tens


# 3. ============== Training the model =======================
print('3. ============== Training the model =======================')
# init the hyperparams
params = OrderedDict(lr=[0.01], batch_size=[100, 1000], num_epochs=[3])
# init the runs
runs = RunBuilder.get_runs(params)

# start the training
print('\n3.1. Starting the training process on {} examples..'.format(len(train_set)))
for run in runs:
    print(run)
    # init the model: we can tune the model hyperparams by adding them to the run and passing them to the constructor
    cnn = CNN()
    # init the optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=run.lr)
    # load the dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=True)
    # tensorboard session
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    tb = SummaryWriter(comment=f'{run}')
    tb.add_image('images', grid)
    tb.add_graph(cnn, images)

    for epoch in range(run.num_epochs):
        total_loss = 0
        total_correct = 0  # correctly predicted images
        accuracy = 0

        for batch in train_loader:
            # get training batch from loader
            images, labels = batch

            # pass the batch through the network
            preds = cnn(images)
            # compute the loss
            loss = F.cross_entropy(preds, labels)

            # reinit the grads to avoid cumulativeness
            optimizer.zero_grad()
            # compute the gradients wrt weights
            loss.backward()
            # update the weights
            optimizer.step()

            total_loss += loss.item()
            total_correct += F.softmax(preds, dim=1).argmax(dim=1).eq(labels).sum().item()

        total_loss /= len(train_set)
        accuracy = total_correct / len(train_set)

        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Number correct', total_correct, epoch)
        tb.add_scalar('accuracy', accuracy, epoch)

        print('epoch: {}, total_correct: {}, loss: {}, accuracy: {}'.format(epoch, total_correct, total_loss, accuracy))

    tb.close()
    print('Run ended..')
print('Training ended..')

# # 4. ============== Testing the model =======================
# print('4. ============== Testing the model =======================')
# test_loss = 0
# test_total_correct = 0
# test_accuracy = 0
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

# print('\n4.1 Starting the testing process on {} examples.'.format(len(test_set)))
# for batch in test_loader:
#     images, labels = batch
#     preds = cnn(images)
#     loss = F.cross_entropy(preds, labels)

#     test_loss += loss.item()
#     test_total_correct += F.softmax(preds, dim=1).argmax(dim=1).eq(labels).sum().item()

# test_loss /= len(test_set)
# test_accuracy = test_total_correct / len(test_set)

# print('Testing ended.. total_correct: {}, loss: {}, accuracy: {}'.format(test_total_correct, test_loss, test_accuracy))
