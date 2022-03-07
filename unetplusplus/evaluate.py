from segmentation_models_pytorch import UnetPlusPlus
from utils import *
from loader import *
from config import *

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from tqdm import tqdm

def visualize(prediction,i,j):
    code2id,id2code,name2id,id2name = one_hot_encoded()
    prediction = prediction.data.cpu().numpy()
    
    prediction = prediction.transpose(1,2,0)
    print(prediction.shape)
    label = onehot_to_rgb(prediction, id2code)
    #print(label.shape)
    plt.imsave(args.output_path+"/{}-{}.png".format(i,j), label)


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
    Model.eval()

    if os.path.exists(args.ModelPath):
        Model.load_state_dict(torch.load(args.ModelPath))
        print("Model Loaded")
    else:
        print("Model not found")

    test_dataset = GTAVDataset(mode="test", dataset_path=args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=2,shuffle=True

    )

    code2id,id2code,name2id,id2name = one_hot_encoded()

    with torch.no_grad():
        for i,data in enumerate(tqdm(test_loader)):
            inputs, labels = data
            labels = [
                rgb_to_onehot(labels[X, :, :, :], id2code)
                for X in range(labels.shape[0])
            ]
            labels = torch.from_numpy(np.asarray(labels))
            inputs, labels = Variable(inputs.cuda()), Variable(
                torch.argmax(labels, -1).cuda()
            )
            outputs = Model(inputs)
            for j in range(labels.shape[0]):
                visualize(outputs[j],i,j)

if __name__ == "__main__":
    main()
