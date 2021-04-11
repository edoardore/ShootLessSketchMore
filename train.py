import argparse
import torch
from DataUtils import load_data
import config
from dataset import SiameseDataset
from torch.utils.data import DataLoader
from models import EmbeddingTUBerlin, EmbeddingMiniQuickDraw
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Few-Shot Learning with Siamese Network')
parser.add_argument('--dataset', type=str, default='miniquickdraw', metavar='N', help='tuberlin/miniquickdraw')
args = parser.parse_args()

print("Training model with " + args.dataset)
if 'tuberlin' == args.dataset:
    net = EmbeddingTUBerlin()
    train_dataset = SiameseDataset(config.TUBerlin + "train.csv", config.TUBerlin)
    val_dataset = SiameseDataset(config.TUBerlin + "val.csv", config.TUBerlin)
    model_name = "tuberlin_model.pt"
elif 'miniquickdraw' == args.dataset:
    net = EmbeddingMiniQuickDraw()
    train_dataset = load_data.QD_Dataset('train')
    val_dataset = load_data.QD_Dataset('test')
    model_name = "miniquickdraw_model.pt"
else:
    raise NameError('Dataset ' + args.dataset + ' not knows')

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

train_losses = []
val_losses = []
train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=2, batch_size=1, shuffle=True)

for epoch in range(config.epochs):
    running_loss = 0.0
    net.train()
    print("Starting epoch " + str(epoch + 1))
    step = 0
    for img1, img2, label in train_dataloader:
        step += 1
        outputs = net(img1, img2)
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("Adam Step " + str(step))
    avg_train_loss = running_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    val_running_loss = 0.0
    # check validation loss after every epoch
    with torch.no_grad():
        net.eval()
        for img1, img2, label in val_dataloader:
            outputs = net(img1, img2)
            loss = criterion(outputs, label)
            val_running_loss += loss.item()
    avg_val_loss = val_running_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)
    print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
          .format(epoch + 1, config.epochs, avg_train_loss, avg_val_loss))
    if avg_train_loss <= min(train_losses):
        torch.save(net.state_dict(), model_name)
        print("Saving best model!")
print("Finished Training")
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Losses')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Losses')
plt.legend()
plt.show()
print(train_losses)
print(val_losses)
