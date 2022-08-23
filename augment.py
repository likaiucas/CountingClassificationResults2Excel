import os
import math
import pandas as pd
import numpy as np
from random import randint, random, sample
from tqdm import tqdm
import cv2


FAIR1M_1_5_CLASSES = ['Airplane', 'Ship', 'Vehicle', 'Basketball_Court', 'Tennis_Court', 
        "Football_Field", "Baseball_Field", 'Intersection', 'Roundabout', 'Bridge'
    ]

config = {
    "max_number":10,
    "random_item_select": True,
    "random_angel":True,
    "random_loc":True,
    'Airplane':0, 'Ship':0, 'Vehicle':0, 'Basketball_Court':5, 'Tennis_Court':5, 
        "Football_Field":5, "Baseball_Field":5, 'Intersection':5, 'Roundabout':5, 'Bridge':5

}

Item_Path = '/usr/local/code/github_code/data/items'
Excl_Path = '/usr/local/code/github_code/JDet/performance.xlsx'
Ori_im_folder = '/usr/local/code/github_code/data/preprocessed/train_1024_200_1.0/images'
Ori_lb_folder = '/usr/local/code/github_code/data/preprocessed/train_1024_200_1.0/labelTxt'

Augim_path = '/usr/local/code/github_code/data/aug'
if not os.path.exists(Augim_path):
    os.mkdir(Augim_path)

Aug_im = os.path.join(Augim_path, 'Augim')
if not os.path.exists(Aug_im):
    os.mkdir(Aug_im)

Aug_lb = os.path.join(Augim_path, 'Auglb')
if not os.path.exists(Aug_lb):
    os.mkdir(Aug_lb)

def forge_img_item(ori_img, itm, img, data, Aug_im=Aug_im, Aug_lb=Aug_lb, config = config):
    def _rand_angel():
        return randint(0,89) # 逆时针旋转角度
        
    def _impath():
        name, _ = os.path.splitext(ori_img['name'])
        return os.path.join(Ori_im_folder, name+'.png')
    def _path(MODE="TXT"):
        name= map(os.path.splitext, items)
        if MODE == "TXT":
            return [os.path.join(Ori_lb_folder, n + '.txt') for n, i in list(name)]
        else:
            return [os.path.join(Ori_im_folder, n + '.png') for n, i in list(name)] 
    
    def _loc(shape1, shape2):
        #  shape1: Item shape
        #  shape2: Zero image
        x,y,c=shape1
        X,Y,C=shape2
        return (randint(x+1, X-x-1), randint(y+1,Y-y-1))

    def _name():
        name= map(os.path.splitext, items)
        return [n for n, i in list(name)]

    def Nrotate(angle,valuex,valuey,pointx,pointy):
        angle = (angle/180)*math.pi
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
        nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
        return (nRotatex, nRotatey)

    def Srotate(angle,valuex,valuey,pointx,pointy):
        angle = (angle/180)*math.pi
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
        sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
        return (sRotatex,sRotatey)

    def rotatecordiate(angle,rectboxs,pointx,pointy):
        output = []
        for rectbox in rectboxs:
            if angle>0:
                output.append(Srotate(angle,rectbox[0],rectbox[1],pointx,pointy))
            else:
                output.append(Nrotate(-angle,rectbox[0],rectbox[1],pointx,pointy))
        return output

    def stick(item_path, image, info, label_txt):
        # info: an excle which store the basic infomation of items. 
        L,W,C = image.shape
        # label_txt = []

        item_path = [os.path.join(Item_Path, itm, i) for i in item_path]
        for it in item_path:
            name = os.path.basename(it)
            loc = info['name']==name
            l, w = info[loc]['l'], info[loc]['w']
            a = _rand_angel()
            # print(it)
            item_image = cv2.imread(it)

            L2, W2, C2 = item_image.shape
            maxLW = max(L2,W2)
            odd = maxLW%2
            item_image=cv2.copyMakeBorder(item_image, 0,maxLW-L2+odd,0, maxLW-W2+odd,cv2.BORDER_CONSTANT,value=[0, 0, 0])
            L2, W2, C2 = item_image.shape
            
            # Rotate the Item
            M = cv2.getRotationMatrix2D((L2//2, W2//2), a, 1)
            M[0,2]+=L2//2
            M[1,2]+=W2//2
            dst = cv2.warpAffine(item_image,M,(2*item_image.shape[0],2*item_image.shape[1]))
            
            L3, W3, C3 = dst.shape
            zeros = np.zeros(image.shape).astype("uint8")
            
            try:
                lc = _loc(dst.shape, image.shape)
            except:
                continue
            zeros[lc[0]:(lc[0]+L3), lc[1]:(lc[1]+W3), :] = dst

            roi = image[lc[0]:(lc[0]+L3+1), lc[1]:(lc[1]+W3+1)] #原始image中，图像的位置
            grayItem = cv2.cvtColor(zeros.astype('uint8'), cv2.COLOR_BGR2GRAY)

            ret, mask = cv2.threshold(grayItem, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)#获取把logo的区域取反 按位运算
            img1_bg = cv2.bitwise_and(image,image,mask = mask_inv)#在img1上面，将logo区域和mask取与使值为0
            img2_fg = cv2.bitwise_and(zeros,zeros,mask = mask)#获取logo的像素信息

            image = cv2.add(img1_bg,img2_fg,dtype=cv2.CV_8U)

            # 计算坐标
            box_origin = np.array([[0.,0.], [0., w.values.tolist()[0]], [l.values.tolist()[0],w.values.tolist()[0]], [l.values.tolist()[0],0.]])
            box = rotatecordiate(a,box_origin,L2//2, W2//2)
            sentence = ''
            for x, y in box:
                sentence=sentence + str(x+M[0,2]+lc[0]) + ' ' + str(y+M[1,2]+lc[1]) + ' '
            sentence = sentence + itm + ' '+'0'+'\n'
            label_txt.append(sentence)
        return image, label_txt

            
        # cv2.imwrite('./test.png', image)

    files= os.listdir(os.path.join(Item_Path, itm)) #输出文件夹下所有贴纸名称
    # img = cv2.imread(_impath()) #等待粘贴的底图
    items = sample(files, randint(0,config[itm]))
    if not items:
        return img, data

    path = _path("PNG") # list(/PATH/TO/IMAGE/FOLDER/img.png)

    info = pd.read_excel(os.path.join(Item_Path, itm + '.xlsx'))
    img2, data = stick(items, img, info, data)

    return img2, data
    
def write_im_txt(img, data, unq, path = Augim_path):
    impath = os.path.join(Aug_im,unq+'.png')
    txtpath = os.path.join(Aug_lb,unq+'.txt')
    cv2.imwrite(impath, img)
    with open(txtpath, 'w') as f:
        for s in data:
            f.write(s)


def main():
    excl_obj=pd.read_excel(Excl_Path)
    loc = excl_obj["Sum"]==0 #筛选条件
    # print(loc)
    Null_items_image = excl_obj[loc] # 没有小目标的数据集
    l = Null_items_image['name'].shape[0]
    for ll in tqdm(range(l)):
        name, _ = os.path.splitext(Null_items_image.iloc[ll]['name'])
        
        img_p = os.path.join(Ori_im_folder, name+'.png')
        img = cv2.imread(img_p)

        txt_p = os.path.join(Ori_lb_folder, name+'.txt')
        sp_name = name.split('-')
        unq = str(ll)+sp_name[0]
        
        data = open(txt_p).readlines()
        for item in FAIR1M_1_5_CLASSES:
            img, data = forge_img_item(Null_items_image.iloc[ll], item, img, data)
        write_im_txt(img, data, unq,path = Augim_path)
        
if __name__=="__main__":
    main()
