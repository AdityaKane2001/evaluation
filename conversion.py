''' Contains classes to convert given annotations/detections to unified annotations format

Annotations format:
dict annotations:
    str identifier
    np.array box (x1,y1,x2,y2)
All boxes assume that each image is 640x640
'''



import numpy as np
import os

class Converter:
    def __init__(self,source,source_type):
        self.source = source #Must be filename or dirname
        self.parse_fn = self.parser
        self.source_type = source_type #Must be `file` or `dir`

    def parser(self):
        '''
        Args:
            None

        Returns:
            `Annotation` object

        Raises:
            Not implemented erroe
        '''

        raise NotImplementedError

    def __call__(self):
        self.parse_fn()

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
    def __init__(self,source):
        Converter.__init__(self,source,source_type='dir')

    def parser(self):
        if os.path.isdir(self.source):
            annotations = dict()
            for i in os.listdir(self.source):
                annotations[i] = self.read_to_arr(os.path.join(self.source,i))
            return annotations
        else:
            raise OSError('Not a directory')

    def read_to_arr(self,file_path):
        annots = []
        with open(file_path,'r') as f:
            for line in f:
                nums = line.split()
                x = float(nums[1]) * 640
                y = float(nums[2]) * 640
                w = float(nums[3]) * 640
                h = float(nums[4]) * 640
                x1 = x - (w/2)
                x2 = x + (w/2)
                y1 = y - (h/2)
                y2 = y + (h/2)
                annots.append([x1,y1,x2,y2])
                del nums,x,y,w,h
        return np.array(annots)
