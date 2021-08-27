# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os


def return_ite(modality, version):
    if version is not None:
        with open (os.path.join(ROOT_DATASET, version, 'actions_label_map.txt'), 'r') as file:
            catigories = file.readlines()
            filename_categories = len(catigories)
        if modality == 'RGB':
            root_data = ROOT_DATASET + 'ite_dataset/frames' 
            filename_imglist_train = f'ite_dataset/{version}/train_videofolder.txt' # contents: path numframes label_num
            filename_imglist_val = f'ite_dataset/{version}/val_videofolder.txt'
            prefix = 'img_{:05d}.jpg' 
        else:
            raise NotImplementedError('no such modality:' + modality)
    else:       
        with open (os.path.join(ROOT_DATASET, 'ite_dataset', 'actions_label_map.txt'), 'r') as file:
            catigories = file.readlines()
            filename_categories = len(catigories)
        if modality == 'RGB':
            root_data = ROOT_DATASET + 'ite_dataset/frames' 
            filename_imglist_train = 'ite_dataset/train_videofolder.txt' # contents: path numframes label_num
            filename_imglist_val = 'ite_dataset/val_videofolder.txt'
            prefix = 'img_{:05d}.jpg'
        else:
            raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ek100(modality, version):
    if version is not None:
        with open (os.path.join(ROOT_DATASET, 'ek-100', version, 'actions_label_map.txt'), 'r') as file:
            catigories = file.readlines()
            filename_categories = len(catigories)
        if modality == 'RGB':
            train_root_data = ROOT_DATASET + 'ek-100/frames_rgb_split/train' 
            val_root_data = ROOT_DATASET + 'ek-100/frames_rgb_split/valid' 
            filename_imglist_train = f'ek-100/{version}/train_videofolder.txt' # contents: path numframes label_num
            filename_imglist_val = f'ek-100/{version}/val_videofolder.txt'
            prefix = 'frame_{:010d}.jpg' 
        else:
            raise NotImplementedError('no such modality:' + modality)
    else:       
        with open (os.path.join(ROOT_DATASET, 'ek-100', 'actions_label_map.txt'), 'r') as file:
            catigories = file.readlines()
            filename_categories = len(catigories)
        if modality == 'RGB':
            train_root_data = ROOT_DATASET + 'ek-100/frames_rgb_split/train' 
            val_root_data = ROOT_DATASET + 'ek-100/frames_rgb_split/valid'
            filename_imglist_train = 'ek-100/train_videofolder.txt' # contents: path numframes label_num
            filename_imglist_val = 'ek-100/val_videofolder.txt'
            prefix = 'frame_{:010d}.jpg'
        else:
            raise NotImplementedError('no such modality:' + modality)
            
    return filename_categories, filename_imglist_train, filename_imglist_val, [train_root_data, val_root_data], prefix

def return_dataset(root, dataset, modality, version = None):
    global ROOT_DATASET
    ROOT_DATASET = root
    dict_single = {'ite': return_ite, 'ek100': return_ek100 }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality, version)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)# /ssd/video/kinetics/labels/train_videofolder.txt
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
