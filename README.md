# Action recoginition starter code, based on [TSM](https://github.com/mit-han-lab/temporal-shift-module)

[[Website]](https://hanlab.mit.edu/projects/tsm/) [[arXiv]](https://arxiv.org/abs/1811.08383)[[Demo]](https://www.youtube.com/watch?v=0T6u7S_gq-4)

```
@inproceedings{lin2019tsm,
  title={TSM: Temporal Shift Module for Efficient Video Understanding},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
} 
```

Thanks to [Yau Weng Kuan](https://github.com/ywk-work) for his contribution to this project.

## 1. Environment setup

install conda 
run 
```conda env create -f environment.yml``` 
or
```'conda create --name ptcpu --file spec_file.txt```

to create 'ptcpu' environment.

activate environment using 
```conda activate ptcpu```
this will create an environment with pytorch cpu version and with all the other libraries used in the repo.  
(Alternatively, install pytorch with GPU support following the official documentation)

## 2. Dataset preparation

Create a actions_label_map.txt file containing the list of actions (classes) that you want to train the model on.  
You need to include a "no_action" class as the negative class.  
Refer to [actions_label_map.txt](https://github.com/benbwf/action_recognition/blob/master/actions_label_map.txt) for an example.  
Place this file in a folder. The name of this folder is your 'dataset version' (you need to specify this when training the model).  
The parent directory of this folder should be in [env_vars.py](https://github.com/benbwf/action_recognition/blob/master/env_vars.py) as your RAW_DATA_ROOT.

## 3. Annotation

Annotate using via_video_annotator.html.  
Create an attribute called 'action', and set the following properties:
 - anchor: 'Temporal Segment in Video or Audio
 - Input Type: Checkbox
Alternatively, load the [annotation_project_template.json](https://github.com/benbwf/action_recognition/blob/master/annotation_project_template.json) file into the annotator to create the action attributes, then edit the action list ad needed.  
Save videos and annotation files into same folder.  
Add that folder path to [env_vars.py](https://github.com/benbwf/action_recognition/blob/master/env_vars.py) as VIDEOS_DIR

## 4. Preprocessing

Set CLIPS_DIR and FRAMES_DIR in [env_vars.py](https://github.com/benbwf/action_recognition/blob/master/env_vars.py) as the locations that you want the preprocessing scripts to save the files to.
Run the three preprocessing scripts/notebooks:
 - [01_preprocess_clip_extractor.ipynb](01_preprocess_clip_extractor.ipynb)
 - [02_preprocess_vid2img.py](./02_preprocess_vid2img.py)
 - [03_preprocess_makedatasetsplit.ipynb](./03_preprocess_makedatasetsplit.ipynb)
 

## 5. Training
download [this checkpoint](https://www.dropbox.com/s/5yxnzubch7b6niu/tsm_rgb.ckpt?dl=1) and place it in ./pretrained

Run the following command to start t raining. Change the necessary parameters (mainly: path to dataset folder, and dataset version)
```
python main.py ite <path/to/to/dataset/folder> RGB --dataset_version <dataset_version> \
--arch resnet50 --num_segments 8 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 25 50 --epochs 100 \
--batch-size 1 -j 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 \
--shift_place=blockres --npb --gpus 0 --tune_from=pretrained/tsm_rgb.ckpt --dense_sample
```
(100 epochs is recommended)

## 6. Prediction
Use [predict_video_and_score.ipynb](https://github.com/benbwf/action_recognition/blob/master/predict_video_and_score.ipynb) to predict actions in the videos.
