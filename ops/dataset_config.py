# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

# ROOT_DATASET = 'C:/Users/Admin/dev/data'  # '/data/jilin/'


# def return_ucf101(modality):
#     filename_categories = 'UCF101/labels/classInd.txt'
#     if modality == 'RGB':
#         root_data = ROOT_DATASET + 'UCF101/jpg'
#         filename_imglist_train = 'UCF101/file_list/ucf101_rgb_train_split_1.txt'
#         filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
#         prefix = 'img_{:05d}.jpg'
#     elif modality == 'Flow':
#         root_data = ROOT_DATASET + 'UCF101/jpg'
#         filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
#         filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
#         prefix = 'flow_{}_{:05d}.jpg'
#     else:
#         raise NotImplementedError('no such modality:' + modality)
#     return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


# def return_hmdb51(modality):
#     filename_categories = 51
#     if modality == 'RGB':
#         root_data = ROOT_DATASET + 'HMDB51/images'
#         filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
#         filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
#         prefix = 'img_{:05d}.jpg'
#     elif modality == 'Flow':
#         root_data = ROOT_DATASET + 'HMDB51/images'
#         filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
#         filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
#         prefix = 'flow_{}_{:05d}.jpg'
#     else:
#         raise NotImplementedError('no such modality:' + modality)
#     return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


# def return_something(modality):
#     filename_categories = 'something/v1/category.txt'
#     if modality == 'RGB':
#         root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
#         filename_imglist_train = 'something/v1/train_videofolder.txt'
#         filename_imglist_val = 'something/v1/val_videofolder.txt'
#         prefix = '{:05d}.jpg'
#     elif modality == 'Flow':
#         root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
#         filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
#         filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
#         prefix = '{:06d}-{}_{:05d}.jpg'
#     else:
#         print('no such modality:'+modality)
#         raise NotImplementedError
#     return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


# def return_somethingv2(modality):
#     filename_categories = 'something/v2/category.txt'
#     if modality == 'RGB':
#         root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
#         filename_imglist_train = 'something/v2/train_videofolder.txt'
#         filename_imglist_val = 'something/v2/val_videofolder.txt'
#         prefix = '{:06d}.jpg'
#     elif modality == 'Flow':
#         root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
#         filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
#         filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
#         prefix = '{:06d}.jpg'
#     else:
#         raise NotImplementedError('no such modality:'+modality)
#     return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


# def return_jester(modality):
#     filename_categories = 'jester/category.txt'
#     if modality == 'RGB':
#         prefix = '{:05d}.jpg'
#         root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
#         filename_imglist_train = 'jester/train_videofolder.txt'
#         filename_imglist_val = 'jester/val_videofolder.txt'
#     else:
#         raise NotImplementedError('no such modality:'+modality)
#     return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality, version):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics400/frames' # /ssd/video/kinetics/images/ #class/video_id/image.jpeg
        filename_imglist_train = 'kinetics400/train_videofolder.txt' # contents: path numframes label_num
        filename_imglist_val = 'kinetics400/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ite(modality, version):
    if version is not None:
        with open (os.path.join(ROOT_DATASET, 'ite_dataset', version, 'actions_label_map.txt'), 'r') as file:
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
    dict_single = {
        # 'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
        #            'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics, 'ite': return_ite, 'ek100': return_ek100 }
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
