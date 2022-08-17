# Put into .../YOUR_JDet_Path/JDet/python/jdet/utils
import os 
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm
from jdet.config.constant import FAIR1M_1_5_CLASSES # I'm Counting the Classes under FAIR1M_1_5 dataset Standard, 
                                                    # it should be change to your standard. 

PATH_TO_ANN = "/usr/local/code/github_code/data/preprocessed/train_1024_200_1.0/labelTxt"

class TxtObj():
    def __init__(self,name) -> None:
        self.name = name
        self.info = {"name":self.name}
        self.info.update(self.getbox())
        self.bboxes=[]
        self.scores=[]
        self.labels=[]
    def getbox(self,):
        ab_path = os.path.join(PATH_TO_ANN, self.name)
        count = {name:0 for name in FAIR1M_1_5_CLASSES}
        datas = open(ab_path).readlines()
        self.bboxes=[]
        self.scores=[]
        self.labels=[]
        for data in datas:
            ds = data.split(" ")
            if len(ds) < 10:
                continue
            self.bboxes.append([float(i) for i in ds[:8]])
            self.scores.append(1)
            self.labels.append(ds[8])
            count[ds[8]]+=1     #START to COUNT
        # if len(self.bboxes) == 0:
        #     self.bboxes = np.zeros([0,8], dtype=np.float32)
        #     self.scores = np.zeros([0], dtype=np.float32)
        #     self.labels = np.zeros([0], dtype=np.int32)
        # else:
        #     self.bboxes = np.array(self.bboxes, dtype=np.float32)
        #     self.scores = np.array(self.scores, dtype=np.float32)
        #     self.labels = np.array(self.labels, dtype=np.int32)
        return count
def ToExcel(AllImg, output_path=''):
    keys = list(AllImg.keys())
    label_keys = list(AllImg[keys[0]].info.keys())

    data = [AllImg[e].info for e in keys]
    pf =pd.DataFrame(data)
    order = label_keys
    pf =pf[order]
    file_path = pd.ExcelWriter(os.path.join(output_path, 'performance.xlsx'))
    pf.to_excel(file_path, sheet_name='pandas', encoding='utf-8', index=False)
    file_path.save()

def main():
    files= os.listdir(PATH_TO_ANN)
    AllTxt = {i:TxtObj(i) for i in tqdm(files)}
    print("Converting data to excel...")
    ToExcel(AllTxt)
    print("Success!")


if __name__=="__main__":
    main()
