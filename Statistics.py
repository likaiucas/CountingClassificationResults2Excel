import os 
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

img_num = 14
Folder_path = 'data'

def get_name(Odnum, mode):
    """
    Return a string, which represents the file name of Object Image. 
    """
    def pstif(num):
        return 'image_'+ str(num)+".tif"
    def pslb(num):
        return 'label_'+ str(num)+".tif"
    def psrgb(num):
        return 'image_'+ str(num)+".png"

    fundict={
        "tif":pstif,
        "label":pslb,
        "rgb":psrgb,
    }
    return os.path.join(Folder_path, fundict[mode](Odnum))

class ImageObj():
    def __init__(self, Odnum=0):
        """ 
        Odnum -> int: the order of image
        """
        self.tif_img=get_name(Odnum, 'tif')
        self.label_img=get_name(Odnum, 'label')
        self.rgb_img=get_name(Odnum, 'rgb')
        self.data_summary = {
            "Image_name":"Img"+str(Odnum),
            10:0,
            11:0,
            12:0,
            20:0,
            21:0,
            22:0,
            23:0,
            30:0,
            40:0,
            50:0,
            60:0,
            61:0,
            62:0,
            71:0,
            72:0,
            73:0,
            74:0,
            80:0,
            90:0,
            100:0,
            255:0,
            0:0,
            "tif_path":self.tif_img if os.path.exists(self.tif_img) else "NOT EXIST!",
            "label_path":self.label_img if os.path.exists(self.label_img) else "NOT EXIST!",
            "rgbRS_path":self.rgb_img if os.path.exists(self.rgb_img) else "NOT EXIST!",
        }
        self.DoStatsitic()

    def DoStatsitic(self,):
        if not os.path.exists(self.label_img):
            print(self.label_img, "NOT EXIST!")
            return False
        label_img = cv2.imread(self.label_img, cv2.IMREAD_GRAYSCALE)
        keys = np.unique(label_img)
        for k in keys:
            self.data_summary[k] = np.sum(label_img==k)/label_img.size     #count the percentage     
        return True
        
def ToExcel(AllImg, output_path=''):
    keys = list(AllImg.keys())
    label_keys = list(AllImg[keys[0]].data_summary.keys())

    data = [AllImg[e].data_summary for e in keys]
    pf =pd.DataFrame(data)
    order = label_keys
    pf =pf[order]
    file_path = pd.ExcelWriter(os.path.join(output_path, 'performance.xlsx'))
    pf.to_excel(file_path, sheet_name='pandas', encoding='utf-8', index=False)
    file_path.save()

def main():
    print("Doing Statistics...")
    AllImg={num:ImageObj(num) for num in tqdm(range(img_num), desc = "Counting for Images:")}
    print("Converting data to excel...")
    ToExcel(AllImg)
    print("Success!")

if __name__=="__main__":
    main()