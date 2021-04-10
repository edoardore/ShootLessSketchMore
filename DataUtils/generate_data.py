import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from random import randint


def generate_dataset(rawdata_root="Data", target_root="MiniQuickDraw", max_samples_per_class=300, show_imgs=False):
    """
    args:
    - rawdata_root: str, specify the directory path of raw data
    - target_root: str, specify the directory path of generated dataset
    - vfold_ratio: float(0-1), specify the test data / total data
    - max_item_per_class: int, specify the max items for each class
        (because the number of items of each class is far more than default value 5000)
    - show_imgs: bool, whether to show some random images after generation done
    """

    # Create the directories for download data and generated dataset.
    if not os.path.isdir(os.path.join("./", rawdata_root)):
        os.makedirs(os.path.join("./", rawdata_root))
    if not os.path.isdir(os.path.join("./", target_root)):
        os.makedirs(os.path.join("./", target_root))

    print("*" * 50)
    print("Generate dataset from npy data")
    print("*" * 50)
    all_files = glob.glob(os.path.join(rawdata_root, '*.npy'))
    print("Classes number: " + str(len(all_files)))
    print("*" * 50)

    # initialize variables
    x = np.empty([0, 784])
    z = np.empty([0, 784])
    y = np.empty([0])
    class_names = []
    class_samples_num = []

    # load each data file
    for idx, file in enumerate(all_files):
        data = np.load(file)
        # print(data.shape)

        indices = np.arange(0, data.shape[0])
        # randomly choose max_items_per_class data from each class
        indices = np.random.choice(
            indices, max_samples_per_class, replace=False)
        data = data[indices]

        # print(data.shape)
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)
        class_samples_num.append(str(data.shape[0]))

        print(str(idx + 1) + "/" + str(len(all_files)) +
              "\t- " + class_name + " has been loaded. \n\t\t Totally " + str(data.shape[0]) + " samples.")
        print("~" * 50)

    print("\n" + "*" * 50)
    print("Data loading done.")

    x_train = x[0:21000, :]
    y_train = y[0:21000]
    w_train = x[0:21000, :]

    x_test = x[21000:x.shape[0], :]
    y_test = y[21000:y.shape[0]]
    w_test = x[21000:x.shape[0], :]

    if show_imgs:
        plt.figure('random images from dataset')
        plt.suptitle('random images from dataset')
        plt.subplot(221)
        idx = randint(0, len(x_train))
        plt.imshow(x_train[idx].reshape(28, 28))
        plt.title(class_names[int(y_train[idx].item())])
        plt.subplot(222)
        idx = randint(0, len(x_train))
        plt.imshow(x_train[idx].reshape(28, 28))
        plt.title(class_names[int(y_train[idx].item())])
        plt.subplot(223)
        idx = randint(0, len(x_train))
        plt.imshow(x_train[idx].reshape(28, 28))
        plt.title(class_names[int(y_train[idx].item())])
        plt.subplot(224)
        idx = randint(0, len(x_train))
        plt.imshow(x_train[idx].reshape(28, 28))
        plt.title(class_names[int(y_train[idx].item())])
        plt.show()

    np.savez_compressed(target_root + "/train", data1=x_train, data2=w_train,
                        target=y_train)

    np.savez_compressed(target_root + "/test", data1=x_test, data2=w_test,
                        target=y_test)

    print("*" * 50)
    print("Great, data_cache has been saved into disk.")
    print("*" * 50)

    with open("./class_names.txt", 'w') as f:
        for i in range(len(class_names)):
            f.write(
                "class name: " + class_names[i] + "\t\tnumber of samples: " + class_samples_num[i] + "\n")

    print("classes_names.txt has been saved.")
    print("*" * 50)


generate_dataset("Data", "MiniQuickDraw", 300, True)
