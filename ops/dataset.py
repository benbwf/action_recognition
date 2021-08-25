# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint

from tools.vid2img_ite import vid2jpg
import glob


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length # 1 if modality is RGB. 5 if modality is flow / rgb diff
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift # if random shift, randomly offset sample indices (get indices). If not random shift, indices are evenly distributed3
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(',') for x in open(self.list_file)] #changed from split(' ') to split(',')
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[-2]) >= 3] # changed from item[1] to item[-2] because some folder names contain spaces
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments 
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments: # this doesn't get triggered?
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        
        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
      
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)
            
        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

class SingleVideoDataLoader(data.IterableDataset):
    """
    dataset class to load samples directly from video
    """


    def __init__(self, path, transform, n_segments, snippet_length, temp_path, frame_rate):
        self.video_path = path
        self.n_segments = n_segments #number of segments/indices to return per clip
        self.snippet_length = snippet_length #length of each clip
        self.video_n_frames = None
        self.video_frame_rate = frame_rate
        self.frames_dir = None
        self.frames_paths = None
        self.transform = transform
        self.temp_path = temp_path
        
        #check if video exists
        if not os.path.exists(self.video_path):
            raise Exception(f'trying to open {self.video_path} but does not exist!')
        
        self.convert_video()
        
        self._parse_list()
        
    def convert_video(self):
        """
        converts video to images, and sets video_n_frames
        """
        video_dir = os.path.dirname(self.video_path)
        video_filename = os.path.basename(self.video_path)
        vid2jpg(video_filename, video_dir, self.temp_path)
        name, ext = os.path.splitext(video_filename)
        self.frames_dir = os.path.join(self.temp_path, name)
        self.frames_paths = glob.glob(os.path.join(self.frames_dir, '*.jpg'))
        self.video_n_frames = len(self.frames_paths)
        print(f'frames_dir: {self.frames_dir}, frames_paths: {self.frames_paths[:5]}\nn_frames: {self.video_n_frames}')

            
    def get_indices(self,
        video_last_frame,
        n_segments=8, #number of segments to return per clip
        video_frame_rate=30, #frame rate of video
        snippet_length = 2 #length of each clip
        ):

        """
        Generates a list of indices to sample frames from a video.
        indices do not overlap, except for last batch if the number of frames do
        not divide equally to fill up the last batch.
        """
        
        
        indices_list = []
        sample_space = video_frame_rate*snippet_length
#         print(video_last_frame, sample_space)
        for i in range(0, int(video_last_frame), int(sample_space)):
            #TODO: return i as the anchor index for reassembling predictions
            start = i
            end = i+sample_space
            if end > video_last_frame:
                end = video_last_frame
                start = video_last_frame - sample_space
            indices = np.linspace(start, end, n_segments, dtype=int)
            indices_list.append(indices)
        return indices_list
            
    def __iter__(self):
       
        for indices in self.indices_list:
            #check if all indices are less than num frames
            if max(indices) <= self.video_n_frames:
                frames, valid, new_indices = self.get_frames_by_frame_pos(frame_pos_list=indices)
#                 frames_valid = [True if frame.all()!=None else False for frame in frames]
                if all(valid):
#                     frames = [Image.fromarray(f) for f in frames]
                    frames = self.transform(frames)
                    #TODO: return i from get_indices as anchor index for reassembling predictions (see get_indices)
                    yield(frames, indices[0])

                elif not all(valid):
                    print(f'error in reading frames! {valid}')
            else:
                print('indices: ',indices, ' video_n_frames: ',self.video_n_frames)
        print('done!', self.video_path)

        
    def _parse_list(self):
        #get indices to load frames
        self.indices_list = self.get_indices(video_last_frame = self.video_n_frames-1,
                                        n_segments = self.n_segments,
                                        video_frame_rate = self.video_frame_rate,
                                        snippet_length = self.snippet_length)
        self.video_list = [VideoRecord([self.video_path, self.snippet_length*self.video_frame_rate, -1]) for item in self.indices_list]

            
    def get_frames_by_frame_pos(self, frame_pos_list):
        frames = []
        valid = []
        new_indices = []
        for frame_pos in frame_pos_list:
            frame_path = self.frames_paths[frame_pos]
            try:
                image = Image.open(frame_path).convert('RGB')
                
                frames.append(image)
                valid.append(True)
            except:
                print(f'unable to open {frame_path}')
                valid.append(False)
        return frames, valid, new_indices

class SlidingWindowDataLoader(SingleVideoDataLoader):
    import cv2
    def __init__(self, path, transform, n_segments, snippet_length, temp_path, frame_rate, stride):
        self.video_path = path
        self.n_segments = n_segments #number of segments/indices to return per clip
        self.snippet_length = snippet_length #length of each clip
        self.video_n_frames = None
        self.video_frame_rate = frame_rate
        self.frames_dir = None
        self.frames_paths = None
        self.transform = transform
        self.temp_path = temp_path
        self.stride = stride
        
        #check if video exists
        if not os.path.exists(self.video_path):
            raise Exception(f'trying to open {self.video_path} but does not exist!')
        
        self.convert_video()
        
        self._parse_list()

    def get_indices(self,
        video_last_frame,
        n_segments=8, #number of segments to return per clip
        video_frame_rate=30, #frame rate of video
        snippet_length = 2, #length of each clip
        stride = 3
        ):

        """
        Generates a list of indices to sample frames from a video.
        if stride = snippet_length*video_frame_rate, then it works the same as non-sliding window data loader
        """


        indices_list = []
        sample_space = video_frame_rate*snippet_length
    #         print(video_last_frame, sample_space)
        #todo: check if video is long enough for at least 1 sample
        anchors=[i for i in range(0, int(video_last_frame), stride)]
        for i in anchors:
            #i is midpoint of sample space
            start = i-sample_space/2
            #need to account for starting of list where sample_space would go into negatives. instead, we just start from 0
            
            start = max(start, 0)
            end = start+sample_space
            #if we reach the end, need to stop advancing end to past our number of frames, and reset start frame accordingly
            if end > video_last_frame:
                end = video_last_frame
                start = video_last_frame - sample_space

            indices = np.linspace(start, end, n_segments, dtype=int)
            indices_list.append(indices)
        return indices_list, anchors

    def _parse_list(self):
        #get indices to load frames
        self.indices_list, self.anchor_list = self.get_indices(video_last_frame = self.video_n_frames-1,
                                        n_segments = self.n_segments,
                                        video_frame_rate = self.video_frame_rate,
                                        snippet_length = self.snippet_length, stride=self.stride)
        self.video_list = [VideoRecord([self.video_path, self.snippet_length*self.video_frame_rate, -1]) for item in self.indices_list]
    def __iter__(self):
       
        for indices, anchor in zip(self.indices_list, self.anchor_list):
            #check if all indices are less than num frames
            if max(indices) <= self.video_n_frames:
                frames, valid, new_indices = self.get_frames_by_frame_pos(frame_pos_list=indices)
#                 frames_valid = [True if frame.all()!=None else False for frame in frames]
                if all(valid):
#                     frames = [Image.fromarray(f) for f in frames]
                    frames = self.transform(frames)
                    #TODO: return i from get_indices as anchor index for reassembling predictions (see get_indices)
                    yield(frames, anchor)

                elif not all(valid):
                    print(f'error in reading frames! {valid}')
            else:
                print('indices: ',indices, ' video_n_frames: ',self.video_n_frames)
        print('done!', self.video_path)




class SingleVideoDataLoaderFromVideo(data.IterableDataset):####DEPRECATED!!!!
    """
    dataset class to load samples directly from video
    """


    def __init__(self, path, transform, n_segments, snippet_length):
        self.video_path = path
        self.n_segments = n_segments #number of segments/indices to return per clip
        self.snippet_length = snippet_length #length of each clip
        self.video_n_frames = None
        
        self.video_frame_rate = None
        self.transform = transform
        
        #check if path exists
        if not os.path.exists(self.video_path):
            raise Exception(f'trying to open {self.video_path} but does not exist!')
                
        self.cap = cv2.VideoCapture(self.video_path) 
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")
            return
        else:
            self.video_n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.video_last_frame = int(self.video_n_frames)-1
            self.video_frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.release()
        
        self._parse_list()
            
            
    def get_indices(self,
        video_last_frame,
        n_segments=8, #number of segments to return per clip
        video_frame_rate=30, #frame rate of video
        snippet_length = 2 #length of each clip
        ):

        """
        Generates a list of indices to sample frames from a video.
        indices do not overlap, except for last batch if the number of frames do
        not divide equally to fill up the last batch.
        """
        
        
        indices_list = []
        sample_space = video_frame_rate*snippet_length
#         print(video_last_frame, sample_space)
        for i in range(0, int(video_last_frame), int(sample_space)):
            start = i
            end = i+sample_space
            if end > video_last_frame:
                end = video_last_frame
                start = video_last_frame - sample_space
            indices = np.linspace(start, end, n_segments, dtype=int)
            indices_list.append(indices)
        return indices_list
            
    def __iter__(self):
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path) 
       
        for indices in self.indices_list:
            #check if all indices are less than num frames
            if max(indices) <= self.video_last_frame:
                frames, valid, new_indices = self.get_frames_by_frame_pos(self.cap, frame_pos_list=indices)
#                 frames_valid = [True if frame.all()!=None else False for frame in frames]
                if all(valid):
                    frames = [Image.fromarray(f) for f in frames]
                    frames = self.transform(frames)
                    yield(frames, indices[0])

                
                elif indices!=new_indices:
                    raise Exception(f'wrong indices returned: {indices},{new_indices},{valid}')
                elif not all(valid):
                    print(f'error in reading frames! {valid}')
            else:
                print('indices: ',indices, ' video_n_frames: ',self.video_n_frames)
        print('done!', self.video_path)
        self.cap.release()
        
    def _parse_list(self):
        #get indices to load frames
        self.indices_list = self.get_indices(video_last_frame = self.video_last_frame,
                                        n_segments = self.n_segments,
                                        video_frame_rate = self.video_frame_rate,
                                        snippet_length = self.snippet_length)
        self.video_list = [VideoRecord([self.video_path, self.snippet_length*self.video_frame_rate, -1]) for item in self.indices_list]

            
    def get_frames_by_frame_pos(self, cap, frame_pos_list):
        frames = []
        valid = []
        new_indices = []
        for frame_pos in frame_pos_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            new_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_indices.append(int(new_frame_pos))
            ret, frame = cap.read()
            frames.append(frame)        
            if ret and float(new_frame_pos) == float(frame_pos):
                valid.append(True)
            else:
                if new_frame_pos!=frame_pos:
                    print(f'unable to set frame position! target frame position: {frame_pos}, result: {new_frame_pos}')
                    valid.append(False)
                elif not ret:
                    print(f'unable to get frame! target frame position: {frame_pos}, video_n_frames: {self.video_n_frames}')
                    valid.append(False)
        return frames, valid, new_indices
