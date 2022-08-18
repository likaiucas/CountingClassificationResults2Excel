import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

# from jdet.config.constant import FAIR1M_1_5_CLASSES #or as follow
FAIR1M_1_5_CLASSES = ['Airplane', 'Ship', 'Vehicle', 'Basketball_Court', 'Tennis_Court', 
        "Football_Field", "Baseball_Field", 'Intersection', 'Roundabout', 'Bridge'
    ]

class Object():
    def __init__(self,):
        self.name = []
        self.l = []
        self.w = []
        self.angle = []
        self.s = []

    def update(self, name, l, w, a):
        self.name.append(name)
        self.l.append(l)
        self.w.append(w)
        self.angle.append(a)
        self.s.append(l*w)

    # def update(self, x):
    #     self.name.extend(x[0])
    #     self.l.extend(x[1])
    #     self.w.extend(x[2])
    #     self.angle.extend(x[3])
    #     self.s.extend(x[1]*x[2])

    def return_dict(self,):
        return {"name":self.name, 
        "l":self.l,
        "w":self.w,
        'angel':self.angle,
        "s":self.s}

# 记录正射图像数据
Obj_dic = {n:Object() for n in FAIR1M_1_5_CLASSES}


EXCEL_HEAD = ['name']+FAIR1M_1_5_CLASSES+['Sum']

Item_path = r"/usr/local/code/github_code/data/items"
Txt_folder = r"/usr/local/code/github_code/data/preprocessed/train_1024_200_1.0/labelTxt"
Img_folder = r"/usr/local/code/github_code/data/preprocessed/train_1024_200_1.0/images"
excel_form = r"/usr/local/code/github_code/JDet/python/jdet/utils/performance.xlsx"

# 为每种类别目标创建文件夹
for n in FAIR1M_1_5_CLASSES:
    if not os.path.exists(os.path.join(Item_path, n)):
        os.mkdir(os.path.join(Item_path, n))

#逆时针旋转
def Nrotate(angle,valuex,valuey,pointx,pointy):
    angle = (angle/180)*math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
    nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
    return (nRotatex, nRotatey)

#顺时针旋转
def Srotate(angle,valuex,valuey,pointx,pointy):
    angle = (angle/180)*math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
    sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
    return (sRotatex,sRotatey)

#将四个点做映射
def rotatecordiate(angle,rectboxs,pointx,pointy):
    output = []
    for rectbox in rectboxs:
        if angle>0:
            output.append(Srotate(angle,rectbox[0],rectbox[1],pointx,pointy))
        else:
            output.append(Nrotate(-angle,rectbox[0],rectbox[1],pointx,pointy))
    return output

# 四个顶点裁剪
def imagecrop(image,box, path_name=''):
    maxx,maxy,_ = image.shape
    xs = [x[1] for x in box]
    ys = [x[0] for x in box]
    # print(xs)
    # print(min(xs),max(xs),min(ys),max(ys))
    cropimage = image[max(0,min(xs)-2):min(max(xs)+2, maxx),max(0, min(ys)-2):min(maxy,max(ys)+2)]
    # print(cropimage.shape)
    # print(path_name)
    cv2.imwrite(path_name,cropimage)
    return cropimage

def CropImage(name):
    # Img_dict = {name:[] for name in FAIR1M_1_5_CLASSES}
    datas = open(os.path.join(Txt_folder, name+'.txt')).readlines()
    image = cv2.imread(os.path.join(Img_folder, name+'.png'))
    for i, data in tqdm(enumerate(datas), desc="processing image:"+name):
        ds = data.split(" ")
        if len(ds) < 10:
            continue
        # 点旋转
        bbox=[int(float(i)) for i in ds[:8]]
        label = ds[8]
        rect = cv2.minAreaRect(np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]], [bbox[6], bbox[7]],]),)#rect为[(旋转中心x坐标，旋转中心y坐标)，(矩形长，矩形宽),旋转角度]
        box_origin = cv2.boxPoints(rect)#box_origin为[(x0,y0),(x1,y1),(x2,y2),(x3,y3)]
        box = rotatecordiate(rect[2],box_origin,rect[0][0],rect[0][1])
        l=name+"_"+label+"_"+str(i)+".png"
        Obj_dic[label].update(l,rect[1][0], rect[1][0], rect[2])

        # 图像旋转
        M = cv2.getRotationMatrix2D(rect[0],rect[2],1)
        dst = cv2.warpAffine(image,M,(2*image.shape[0],2*image.shape[1]))
        imagecrop(dst,np.int0(box), path_name = os.path.join(Item_path, label, name+"-"+label+"_"+str(i)+".png"))

def Toexcel(dk, output_path=Item_path):
    data = Obj_dic[dk].return_dict()
    label_keys = list(data.keys())

    pf =pd.DataFrame()
    for l in label_keys:
        pf[l] = data[l]
    
    # order = label_keys
    # pf =pf[order]
    file_path = pd.ExcelWriter(os.path.join(output_path, dk,dk+'.xlsx'))
    pf.to_excel(file_path, sheet_name='pandas', encoding='utf-8', index=False)
    file_path.save()

def main():
    # 开始逐文件处理
    excl_obj=pd.read_excel(excel_form)
    print('Processing clip items....')
    for i, n in enumerate(excl_obj['name']):
        name, _ = os.path.splitext(n)
        CropImage(name)

    # 记录数据并汇总
    for item in tqdm(FAIR1M_1_5_CLASSES):
        Toexcel(item)

if __name__=="__main__":
    main()