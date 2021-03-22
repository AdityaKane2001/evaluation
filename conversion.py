''' Contains classes to convert given annotations/detections to unified annotations format

Annotations format:
dict annotations:
    str identifier
    np.array box (x1,y1,x2,y2)
All boxes assume that each image is 640x640
'''

import numpy as np
import pandas as pd
import os


class Converter:
    def __init__(self, source, source_type):
        self.source = source  # Must be filename or dirname
        self.parse_fn = self.parser
        self.source_type = source_type  # Must be `file` or `dir`

    def parser(self):
        '''
        Args:
            None

        Returns:
            `Annotation` object

        Raises:
            Not implemented error
        '''

        raise NotImplementedError

    def __call__(self):
        return self.parse_fn()


'''
class Annotation:
    def __init__(self,annotations=None):
        ''''''
        annotations:
            list of tuples having (identifier,annots)
            identifier: filename
            annots: (x1,y1,x2,y2)
        ''''''

        self.annotations = []
        if annotations is not None:
            for i in annotations:
                annots = dict()
                annots[i[0]] = np.array(i[1])
                self.annotations.append(annots)
            self.df = self.to_df()

    def to_df(self):
        return pd.DataFrame(self.annotations)

    def to_csv(self,csv_path):
        self.df.to_csv(csv_path,index=False)
'''


class YOLOConverter(Converter):
    def __init__(self, source):
        Converter.__init__(self, source, source_type='dir')

    def parser(self):
        if os.path.isdir(self.source):
            annotations = dict()
            for i in os.listdir(self.source):
                annotations[i] = self.read_to_arr(os.path.join(self.source, i))
            return annotations
        else:
            raise OSError('Not a directory')

    def read_to_arr(self, file_path):
        annots = []
        with open(file_path, 'r') as f:
            for line in f:
                nums = line.split()
                x1 = float(nums[1])   * 640
                y1 = float(nums[2])   * 640
                x2 = float(nums[3])   * 640
                y2 = float(nums[4])   * 640
                # x1 = x - (w/2)
                # x2 = x + (w/2)
                # y1 = y - (h/2)
                # y2 = y + (h/2)
                annots.append([x1, y1, x2, y2])
                # del nums,x,y,w,h
        return np.array(annots)


class RetinaConverter(Converter):
    def __init__(self, source, skip=False):
        Converter.__init__(self, source, source_type='file')
        self.skip = skip


    def parser(self):
        if self.skip == True:
            df = pd.read_csv(self.source, header=None, skiprows=15)
        else:
            df = pd.read_csv(self.source, header=None)
        img_names = list(set(df[0]))
        annotations = dict()
        for i in img_names:
            part = df.loc[df[0] == i]

            if len(part) > 1:
                annots = []

                for j in part.iterrows():
                    x1 = float(j[1][1])   * 640 / 416
                    y1 = float(j[1][2])   * 640 / 416
                    x2 = float(j[1][3])   * 640 / 416
                    y2 = float(j[1][4])   * 640 / 416
                    conf = float(j[1][5])
                    annots.append([x1, y1, x2, y2, conf])
                annots = np.array(annots)

                annotations[i.split('/')[3]] = annots
            else:
                annots = []

                x1 = float(part[1])   * 640 / 416
                y1 = float(part[2])   * 640 / 416
                x2 = float(part[3])   * 640 / 416
                y2 = float(part[4])   * 640 / 416
                conf = float(part[5])
                annots.append([x1, y1, x2, y2, conf])
                annotations[i.split('/')[3]] = np.array(annots)

        return annotations

# TODO CRAFTConverter


class CRAFTConverter(Converter):
    def __init__(self, source):
        Converter.__init__(self, source, source_type='dir')
        self.source = source

    def parser(self):
        annotations = dict()
        annot_list = self.get_annot_list()
        for i in annot_list:
            annots = self.read_to_arr('craft_result/'+i)
            annotations[i[4:]] = np.array(annots)
        return annotations




    def read_to_arr(self, file_path):
        annots = []
        with open(file_path, 'r') as f:
            for line in f:
                if line=='':
                    continue

                nums = line.split(',')
                x1 = float(nums[0]) * 640 /416
                y1 = float(nums[1]) * 640 /416
                x2 = float(nums[2]) * 640 /416
                y2 = float(nums[3]) * 640 /416

                x3 = float(nums[4]) * 640 /416
                y3 = float(nums[5]) * 640 /416
                x4 = float(nums[6]) * 640 /416
                y4 = float(nums[7]) * 640 /416
                annots.append([x1, y1, x2, y2,x3,y3,x4,y4])
                # del nums,x,y,w,h
        return np.array(annots)




    def get_annot_list(self):
        all_list = os.listdir('craft_result')
        annot_list = [i for i in all_list if i.endswith('.txt')]
        return annot_list

class CRAFTEvaluator:
    #TODO
    def __init__(self):
        pass