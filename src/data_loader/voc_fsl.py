import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse
import json
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
# from utils.util import *
import random
# import zipfile38 as zipfile
import pickle

# import util
# from util import *

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels(root, dataset, subset):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + subset + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(len(object_categories)):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):

    unlabeled = 0
    remaining = 0
    total = 0
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                total += 1
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                if labels.sum() <= 0:
                    unlabeled += 1
                else:
                    remaining += 1
                    item = (name, labels)
                    images.append(item)
            rownum += 1
    print("Total : " + str(total))
    print("Unlabeled : " + str(unlabeled))
    print("Remaining : " + str(remaining))
    return images, True if num_categories == 14 else False


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images


def download_voc2007(root):
    pass


class Voc2007Classification(data.Dataset):
    has_selected = []
    def __init__(self, root, subset, transform=None, target_transform=None, inp_name=None):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.annotations = os.path.join(self.root, 'files', 'VOC2007')
        self.set = subset
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        download_voc2007(self.root)

        # define path of csv file
        # path_csv = os.path.join(self.root, 'files', 'VOC2007')
        # define filename of csv file
        file_csv = os.path.join(self.annotations, 'classification_' + subset + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            # if not os.path.exists(path_csv):  # create dir if necessary
            #     os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2007', self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
            subset, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path, self.inp), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)



class Voc2007Classification_fsl(data.Dataset):
    has_selected = []
    def __init__(self, root, file_csv, transform=None, target_transform=None, word_emb_file=None):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root,  'VOCdevkit','VOC2007','JPEGImages')
        self.annotation = os.path.join(root, 'files', 'VOC2007')
        self.transform = transform
        self.target_transform = target_transform

        random.seed(0)
        self.novel_cls_ind = random.sample(range(len(object_categories)), 6)
        self.novel_cls_ind.sort()
        self.base_cls_ind = list(set(range(len(object_categories))) - set(self.novel_cls_ind))
        self.base_cls_ind.sort()

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            # # generate csv file
            # labeled_data = read_object_labels(self.root, self.set)
            # # write csv file
            # write_object_labels_csv(file_csv, labeled_data)
            print("no partition csv file:",file_csv)
            raise NotImplementedError
        #
        self.classes = object_categories
        self.images, is_base = read_object_labels_csv(file_csv)

        with open(word_emb_file, 'rb') as f:
            self.inp = pickle.load(f).astype('float32')
        if is_base:
            self.inp = self.inp[self.base_cls_ind]
        else:
            self.inp = self.inp[self.novel_cls_ind]

        self.inp_name = word_emb_file

        print('[dataset] VOC2007 classification set=%s number of classes=%d  number of images=%d' % (
            file_csv, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path.zfill(6) + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.inp

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        print(self.__len__())
        return len(self.classes)


# if __name__ == '__main__':
#     read_object_labels_csv('/data2/yanjiexuan/voc/files/VOC2007/classification_trainval.csv')