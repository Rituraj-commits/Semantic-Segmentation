from utils import *
from loader import *
from config import *

from segmentation_models_pytorch import UnetPlusPlus

import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

import warnings

warnings.filterwarnings("ignore")


def main():
    Model = UnetPlusPlus(
        encoder_name="efficientnet-b1",
        encoder_depth=5,
        encoder_weights=None,
        in_channels=3,
        classes=35,
        activation=None,
    )
    Model.cuda()

    train_dataset = GTAVDataset(mode="train", dataset_path=args.dataset_path)
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(args.validation_split * len(train_dataset)))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=2, sampler=train_sampler
    )
    val_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=2, sampler=valid_sampler
    )

    if args.optimizer == "adam":
        optimizer = optim.Adam(Model.parameters(), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, mode="min", patience=5, verbose=True, factor=0.5
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(Model.parameters(), lr=args.learning_rate, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    best_Model = 1.0
    tb = SummaryWriter()

    code2id,id2code,name2id,id2name = one_hot_encoded()

    if not os.path.isdir(args.ModelSavePath):
        os.makedirs(args.ModelSavePath)

    print("Start Training")

    for epoch in tqdm(range(args.epochs)):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data
            labels = [rgb_to_onehot(labels[X,:,:,:], id2code) for X in range(labels.shape[0])]
            labels = torch.squeeze(torch.tensor(np.array(labels),dtype=torch.long),dim=1)
            inputs, labels = Variable(inputs.cuda()), Variable(torch.max(labels,1)[1].cuda())
            outputs = Model(inputs)
            print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(
                    "Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f"
                    % (epoch + 1, args.epochs, i + 1, len(train_loader), loss.item())
                )

        if (epoch + 1) % 10 == 0:
            Model.eval()
            val_loss = 0.0
            val_dice = 0.0

            for i, data in enumerate(val_loader):
                inputs, labels = data
                labels = [rgb_to_onehot(labels[X,:,:,:], id2code) for X in range(labels.shape[0])]
                labels = torch.squeeze(torch.tensor(np.array(labels),dtype=torch.long),dim=1)
                inputs, labels = Variable(inputs.cuda()), Variable(torch.max(labels,1)[1].cuda())
                outputs = Model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_dice += dice_score(outputs, labels)
            val_loss /= len(val_loader)
            val_dice /= len(val_loader)

            with open("VAL_LOGS.txt", "a+") as f:
                f.write("epoch: %s," % str(epoch + 1))
                f.write("val_dice: %s\n" % str(val_dice))
               

            print("Validation Dice: %.4f, Validation Loss" % (val_dice, val_loss))

            tb.add_scalar("Validation Dice", val_dice, epoch)
            tb.add_scalar("Validation Loss", val_loss, epoch)
            Model.train()
            if val_loss < best_Model:
                best_Model = val_loss
                torch.save(Model.state_dict(), args.ModelSavePath + "best_Model.pkl")
                print("Saving best Model")

        scheduler.step(val_loss)


if __name__ == "__main__":
    main()
