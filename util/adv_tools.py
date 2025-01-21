import copy

import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
import numpy as np

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescentPyTorch,
    SquareAttack,
    SaliencyMapMethod,
    DeepFool,
    ElasticNet,
    CarliniL2Method,
)

# from models import LeNet
# from compressions import jpegRGB

MAX_LENGTH = 500
BATCH_SIZE = 10
CLASS_COUNT = 1000


def get_attack(
    data,
    aname,
    strength,
    model: nn.Module,
    input_shape=(1, 28, 28),
    batch_size=BATCH_SIZE,
):
    # Step 2a: Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)

    data = copy.deepcopy(data)

    # Step 3: Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=CLASS_COUNT,
    )

    # Step 6: Generate adversarial test examplesck (Black Box)  = 4/255, queries = 1000 9.29 36.47 (+27.18) 73.79 (+64.50) 71.18 (+61.89)
    if aname == "FGSM":
        attack = FastGradientMethod(
            estimator=classifier, eps=strength, batch_size=batch_size
        )
    elif aname == "PGD":
        attack = ProjectedGradientDescentPyTorch(
            estimator=classifier,
            max_iter=30,
            eps=strength,
            norm="inf",
            batch_size=batch_size,
        )
    elif aname == "SQA":
        attack = SquareAttack(
            estimator=classifier,
            max_iter=1000,
            eps=strength,
            norm="inf",
            batch_size=batch_size,
        )
    elif aname == "JSMA":
        attack = SaliencyMapMethod(classifier=classifier, batch_size=batch_size)
    elif aname == "DeepFool":
        attack = DeepFool(
            classifier=classifier,
            epsilon=strength,
            max_iter=1000,
            batch_size=batch_size,
        )
    elif aname == "ElasticNet":
        attack = ElasticNet(classifier=classifier, max_iter=1000, batch_size=batch_size)
    elif aname == "CW":
        attack = CarliniL2Method(
            classifier=classifier,
            binary_search_steps=9,
            max_iter=1000,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Invalid attack name: {aname}")

    # pick random 5 images and save
    # for i in range(5):
    #     plt.imshow(np_to_pil(data[i]))
    #     plt.savefig(f'img/clean_{i}.png')

    x_adv = np.array([])
    for i in range(0, len(data), MAX_LENGTH):
        if i == 0:
            x_adv = attack.generate(x=data[i : i + MAX_LENGTH])
        else:
            x_adv = np.concatenate(
                (x_adv, attack.generate(x=data[i : i + MAX_LENGTH])), axis=0
            )

    # # pick random 5 images and save
    # for i in range(5):
    #     plt.imshow(np_to_pil(x_adv[i]))
    #     plt.savefig(f'img/adv_{aname}_{i}.png')
    # x_adv = (torch.tensor(i, dtype=torch.float32) for i in x_adv)

    return x_adv


def loader_to_data(loader: DataLoader):
    # get x_test from test_loader
    data = [[], []]
    test_raw_loader_copy = copy.deepcopy(loader)

    for i, batch in enumerate(test_raw_loader_copy):
        # if i > 10:
        #     break
        data[0] += batch[0]
        data[1] += batch[1]

    data_len = len(data[1])
    x_test = np.array(
        [data[0][i].detach().numpy() for i in range(data_len)], dtype="float32"
    )
    y_test = np.array([data[1][i] for i in range(data_len)], dtype="int")

    return x_test, y_test


def data_to_loader(x, Y, batch_size=64, should_shuffle=False):
    x = (torch.tensor(i, dtype=torch.float32) for i in x)
    x = tuple(x)
    Y = (torch.tensor(i, dtype=torch.uint8) for i in Y)
    Y = tuple(Y)
    result = TensorDataset(torch.stack(x), torch.stack(Y))
    return DataLoader(result, batch_size=batch_size, shuffle=should_shuffle)  # noqa


# def compress_dataset(dataset, length, name, quality=75):
#     result = []

#     for i in range(length):
#         img = np_to_pil(dataset[i])

#         jpeg_t = jpegRGB(quality)
#         img = jpeg_t(img)

#         tensor_t = ToTensor()
#         img = tensor_t(img).numpy()

#         # # Assuming dataset[i] and img are your images
#         # fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create 2 subplots side by side

#         # # Show the first image in the first subplot
#         # axs[0].imshow(dataset[i].transpose(1, 2, 0))
#         # axs[0].set_title('Original Image')

#         # # Show the second image in the second subplot
#         # axs[1].imshow(img.transpose(1, 2, 0))
#         # axs[1].set_title('Transformed Image')

#         # fig.show()

#         # exit()

#         # if i < 5:
#         #     print(img.shape)
#         #     plt.imshow(img.transpose(1, 2, 0))
#         #     plt.savefig(f'img/adv_{name}_jpg_{i}.png')

#         result.append(img)

#     return result


def np_to_pil(in_array):
    pil_transform = ToPILImage()
    return pil_transform(torch.tensor(in_array))
