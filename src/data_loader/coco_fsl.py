"""
Pytorch datasets for loading COCO.
"""
from collections import defaultdict
import itertools
import logging
import os
import pickle as pkl
import random

from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import trange
import torchvision.transforms as transforms
import clip

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

COCO_CLASS_NUM = 80


def labels_list_to_1hot(labels_list, class_list):
    """Convert a list of indices to 1hot representation."""

    labels_1hot = np.zeros(COCO_CLASS_NUM, dtype=np.float32)

    labels_1hot[list(filter(lambda x: x in class_list, labels_list))] = 1
    assert labels_1hot.sum() > 0, "No labels in conversion of labels list to 1hot"

    return labels_1hot


def load_image(img_path):
    img = Image.open(img_path)

    if len(np.array(img).shape) == 2:
        img = img.convert('RGB')

    return img


class CocoDataset(Dataset):
    def __init__(
            self,
            root_dir,
            set_name='train2014',
            unseen_set=False,
            transform=None,
            return_ids=False,
            inp_name= None,
            debug_size=-1):
        """COCO Dataset
        Args:
            root_dir (string): COCO directory.
            set_name (string): 'train2014'/'val2014'.
            unseen_set (bool): Whether to use the seen (64 classes) or unseen (16) classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            return_ids (bool, optional): Whether to return also the image ids.
            debug_size (int, optional): Subsample the dataset. Useful for debug.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.unseen_set = unseen_set
        self.return_ids = return_ids
        self.deubg_size = debug_size

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        with open(inp_name, 'rb') as f:
            self.semantic = pkl.load(f)
        self.inp_name = inp_name

        self.setup_classes()
        self.load_classes()
        self.calc_indices()

    def setup_classes(self):
        """Setup the seen/unseen classes and labels/images mappings."""

        #
        # Create a random (up to seed) set of (64/16 out of 80) classes.
        #
        random.seed(0)
        self.labels_list = set(random.sample(range(80), 64))
        if self.unseen_set:
            self.labels_list = set(range(80)) - self.labels_list
        self.labels_list = sorted(list(self.labels_list))
        self.semantic = self.semantic[self.labels_list]

    def load_classes(self):
        """Load class/categories/labels and create mapping."""

        #
        # Load class names (name -> label)
        #
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.label_to_category = {i: c['id'] for i, c in enumerate(categories)}
        self.category_to_label = {v: k for k, v in self.label_to_category.items()}
        self.label_to_class = {i: c['name'] for i, c in enumerate(categories)}
        self.class_to_label = {v: k for k, v in self.label_to_class.items()}

    def calc_indices(self):
        """Setup the filtered images lists."""
        #
        # Map classes/labels to image indices.
        #
        labels_to_img_ids = defaultdict(list)
        for image_id in self.image_ids:
            labels = set(self.load_labels(image_id))

            #
            # Filter images with no labels.
            #
            if not labels:
                continue

            #
            # Map labels to their images (ids).
            #
            for label in labels:
                labels_to_img_ids[label].append(image_id)

        #
        # Filter images according to labels list.
        #
        image_ids_lists = [labels_to_img_ids[l] for l in self.labels_list]
        self.image_ids = list(set(itertools.chain(*image_ids_lists)))

        #
        # Create the image paths and labels list (for the dataset __getitem__ method).
        #
        self.image_paths = []
        self.image_labels = []
        for image_id in self.image_ids:
            self.image_paths.append(self.image_id_to_path(image_id))
            self.image_labels.append(
                labels_list_to_1hot(self.load_labels(image_id), self.labels_list)
            )

    def load_labels(self, image_id):
        """Get the labels of an image by image_id."""

        #
        # get ground truth annotations
        #
        annotations_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)

        #
        # some images appear to miss annotations (like image with id 257034)
        #
        if len(annotations_ids) == 0:
            return []

        #
        # parse annotations
        #
        coco_annotations = self.coco.loadAnns(annotations_ids)
        labels = [self.category_to_label[ca['category_id']] \
                       for ca in coco_annotations if ca['bbox'][2] > 0 and ca['bbox'][3] > 0]

        return sorted(set(labels))

    def image_id_to_path(self, img_id):
        """Convert image ids to paths."""

        #
        # Note:
        # The reason I use `int` is that `img_id` might be numpy scalar, and coco
        # doesn't like it.
        #
        image_info = self.coco.loadImgs(int(img_id))[0]
        return os.path.join(self.root_dir, self.set_name, image_info['file_name'])

    def __len__(self):

        if self.deubg_size > 0:
            return self.deubg_size

        return len(self.image_ids)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        labels = self.image_labels[idx]
        labels = labels[self.labels_list]
        
        if self.transform:
            img = self.transform(img)

        if self.return_ids:
            return img, labels, self.image_ids[idx]

        return img, labels, self.semantic

class CocoDatasetAugmentation(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, used_ind_path, class_ind_dict_path, set_name='train2014', transform=None, inp_name = None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        random.seed(0)
        tmp = random.sample(range(80), 64)
        tmp.sort()
        self.classListInv = tmp
        classList = [idx for idx in range(80) if idx not in tmp]
        classList.sort()
        print("Class list: ", classList)
        self.classList = classList

        with open(inp_name, 'rb') as f:
            self.semantic = pkl.load(f)
        self.inp_name = inp_name

        self.semantic = self.semantic[self.classList]

        self.load_classes()
        self.classes_num = 80
        self.class_histogram = np.zeros(80)
        # all 80 classes
        ClassIdxDict80 = {el: [] for el in range(80)}

        one_label = 0
        class16Indices = []
        for idx in range(len(self.image_ids)):
            labels = self.load_annotations(idx)
            if len(labels) == 1:
                one_label += 1
            if not labels:
                continue
            filteredLabels = [label for label in labels if label in self.classList]
            if len(filteredLabels) > 0:
                class16Indices += [idx]
            for label in set(labels):
                ClassIdxDict80[label] += [idx]
        self.class16Indices = class16Indices
        self.ClassIdxDict80 = ClassIdxDict80
        with open(class_ind_dict_path, 'rb') as f:
            ClassIdxDict16 = pkl.load(f)
        for key in ClassIdxDict16.keys():
            print("values number for key {}: {}".format(key, len(ClassIdxDict16[key])))
        print("ClassIdxDict16 keys: ", ClassIdxDict16.keys())
        print("ClassIdxDict16 values: ", ClassIdxDict16.values())
        self.ClassIdxDict16 = ClassIdxDict16
        print("img num: ", len(self.image_ids))
        with open(used_ind_path, 'rb') as f:
            usedIndices = pkl.load(f)
        self.usedIndices = usedIndices
        print("used indices len: ", len(usedIndices))
        print("used indices: ", usedIndices)
        self.class_histogram = np.zeros(80)
        self.classes_num = 80

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        self.label_to_category = {i: c['id'] for i, c in enumerate(categories)}
        self.category_to_label = {v: k for k, v in self.label_to_category.items()}
        self.label_to_class = {i: c['name'] for i, c in enumerate(categories)}
        self.class_to_label = {v: k for k, v in self.label_to_class.items()}
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        if self.set_name == 'train2014':
            res = len(self.usedIndices)
        else:
            res = len(self.image_ids)
        return res

    def __getitem__(self, idx):
        found = False
        while not found:
            tmp = np.random.randint(len(self.classList))
            min_label = self.classList[tmp]
            for ind in range(self.classes_num):
                if ind in self.classList:
                    if self.class_histogram[ind] < self.class_histogram[min_label]:
                        min_label = ind
            class_name1 = min_label
            if self.set_name == 'val2014':
                tmpIdx1 = idx % len(self.ClassIdxDict80[class_name1])
                idx1 = self.ClassIdxDict80[class_name1][tmpIdx1]
                labels1 = self.load_annotations(idx1)
            else:
                tmpIdx1 = idx % len(self.ClassIdxDict16[class_name1])
                idx1 = self.ClassIdxDict16[class_name1][tmpIdx1]
                labels1 = self.load_annotations(idx1)
            labels1 = list(set(labels1))
            filteredLabels = [label for label in labels1 if label in self.classList]
            if len(filteredLabels) == 0:
                idx = np.random.randint(self.__len__())
            else:
                found = True
        
        labels = labels_list_to_1hot(filteredLabels, self.classList)
        one_indices = np.where(labels == 1)
        one_indices = one_indices[0]
        self.class_histogram[one_indices] += 1
        img = self.load_image(idx1)
        if self.transform:
            img = self.transform(img)
        
        labels = labels_list_to_1hot(filteredLabels, self.classList)
        labels = labels[self.classList]

        torLab = torch.from_numpy(labels)
        return img, torLab, self.semantic

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = Image.open(path)

        if len(np.array(img).shape) == 2:
            img = img.convert('RGB')

        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = []

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            annotations += [self.coco_label_to_label(a['category_id'])]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80

class CocoDatasetFsl(Dataset):
    def __init__(self, root_dir, used_ind_path, class_ind_dict_path, set_name='train2014', transform=None):
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        random.seed(0)
        tmp = random.sample(range(80), 64)
        tmp.sort()
        self.classListInv = tmp
        classList = [idx for idx in range(80) if idx not in tmp]
        classList.sort()
        assert len(classList) == 16

        self.classList = classList
        self.load_classes()


    def load_classes(self):
        """Load class/categories/labels and create mapping."""

        #
        # Load class names (name -> label)
        #
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.label_to_category = {i: c['id'] for i, c in enumerate(categories)}
        self.category_to_label = {v: k for k, v in self.label_to_category.items()}
        self.label_to_class = {i: c['name'] for i, c in enumerate(categories)}
        self.class_to_label = {v: k for k, v in self.label_to_class.items()}

class CocoCILP(CocoDataset):
    def __init__(self, root_dir, set_name = 'train2014', unseen_set=False, transform=None, return_ids=False, inp_name= None, debug_size=-1):
        super().__init__(root_dir, set_name = set_name, unseen_set=unseen_set, transform=transform, return_ids=return_ids, inp_name= inp_name, debug_size=-1)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        labels = self.image_labels[idx]

        labels = labels[self.labels_list]
        
        if self.transform:
            img = self.transform(img)

        if self.return_ids:
            return img, labels, self.image_ids[idx]

        return img, labels, self.semantic   

    
class CocoFslCLIP(CocoDatasetAugmentation):
    def __init__(self, root_dir, used_ind_path, class_ind_dict_path, set_name='train2014', transform=None, inp_name = None):
        super().__init__(root_dir, used_ind_path, class_ind_dict_path, set_name=set_name, transform=transform, inp_name = inp_name)

    def __getitem__(self, idx):
        found = False
        while not found:
            tmp = np.random.randint(len(self.classList))
            min_label = self.classList[tmp]
            for ind in range(self.classes_num):
                if ind in self.classList:
                    if self.class_histogram[ind] < self.class_histogram[min_label]:
                        min_label = ind
            class_name1 = min_label
            if self.set_name == 'val2014':
                tmpIdx1 = idx % len(self.ClassIdxDict80[class_name1])
                idx1 = self.ClassIdxDict80[class_name1][tmpIdx1]
                labels1 = self.load_annotations(idx1)
            else:
                tmpIdx1 = idx % len(self.ClassIdxDict16[class_name1])
                idx1 = self.ClassIdxDict16[class_name1][tmpIdx1]
                labels1 = self.load_annotations(idx1)
            labels1 = list(set(labels1))
            filteredLabels = [label for label in labels1 if label in self.classList]
            if len(filteredLabels) == 0:
                idx = np.random.randint(self.__len__())
            else:
                found = True
        
        labels = labels_list_to_1hot(filteredLabels, self.classList)
        one_indices = np.where(labels == 1)
        one_indices = one_indices[0]
        self.class_histogram[one_indices] += 1
        img = self.load_image(idx1)
        if self.transform:
            img = self.transform(img)
        
        labels = labels_list_to_1hot(filteredLabels, self.classList)
        labels = labels[self.classList]

        torLab = torch.from_numpy(labels)
        return img, torLab, self.semantic