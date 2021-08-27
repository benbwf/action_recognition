# Action recoginition, based on [TSM](https://github.com/mit-han-lab/temporal-shift-module)

[[Website]](https://hanlab.mit.edu/projects/tsm/) [[arXiv]](https://arxiv.org/abs/1811.08383)[[Demo]](https://www.youtube.com/watch?v=0T6u7S_gq-4)

```
@inproceedings{lin2019tsm,
  title={TSM: Temporal Shift Module for Efficient Video Understanding},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
} 
```

## 0. Environment setup

install conda 
run 'conda env create -f environment.yml' to create 'ptcpu' environment.
or
'conda create --name <env_name> --file spec_file.txt

activate environment using 'conda activate ptcpu'
this will create an environment with pytorch cpu version
(alternatively, install pytorch with GPU support following the official documentation)

## Dataset preparation

Create a actions_label_map.txt file containing the list of actions (classes) that you want to train the model on. 
You need to include a "no_action" class as the negative class.
Refer to actions_label_map.txt for an example.
Place this file in a folder. The name of this folder is your 'dataset version' (you need to specify this when training the model).
The parent directory of this folder should be in env_vars.py as your RAW_DATA_ROOT.

## 1. Annotation

Annotate using via_video_annotator.html.
Save videos and annotation files into folder.
Add folder to env_vars.py as VIDEOS_DIR

## 2. Preprocessing

Set CLIPS_DIR and FRAMES_DIR in env_vars.py as the locations that you want the preprocessing scripts to save the files to.
Run the three preprocessing scripts/notebooks

## 3. Training
download [this checkpoint](https://www.dropbox.com/s/5yxnzubch7b6niu/tsm_rgb.ckpt?dl=1) and place it in ./pretrained


python main.py ite <path/to/to/dataset/folder> RGB --dataset_version <dataset_version> --arch resnet50 --num_segments 8 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 25 50 --epochs 100 --batch-size 1 -j 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb --gpus 0 --tune_from=pretrained/tsm_rgb.ckpt --dense_sample

(100 epochs is recommended)
