''' Contains classes to convert given annotations/detections to unified annotations format

Annotations format:
dict annotations:
    str identifier
    np.array box (x1,y1,x2,y2)
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
        else:
            raise OSError('Not a directory')

    def read_to_arr(self,file_path):
        annots = []
        with open(file_path,'r') as f:
            for line in f:
                nums = line.split()
                annots.append(nums[1:-1])
        return np.array(annots)