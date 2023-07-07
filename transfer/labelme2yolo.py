# trans_labelme_to_yolo.py

import cv2
import os
import json
import shutil
import numpy as np
from pathlib import Path
from glob import glob
import re
import math
id2cls = {1: 'mouse'}
cls2id = {'mouse': 1}
 
#支持中文路径
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),flags=cv2.IMREAD_COLOR)
    return cv_img
 
def labelme2yolo_single(img_path,label_file):
    anno= json.load(open(label_file, "r", encoding="utf-8"))
    shapes = anno['shapes']
    w0, h0 = anno['imageWidth'], anno['imageHeight']
    image_path = os.path.basename(img_path + anno['imagePath'])
    labels = []
    for s in shapes:
        pts = s['points']
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x = (x1 + x2) / 2 / w0 
        y = (y1 + y2) / 2 / h0
        w  = abs(x2 - x1) / w0
        h  = abs(y2 - y1) / h0
        cid = cls2id[s['label']]    
        labels.append([cid, x, y, w, h])
    return np.array(labels), image_path
 
def labelme2yolo(img_path,labelme_label_dir, save_dir='res/'):
    labelme_label_dir = str(Path(labelme_label_dir)) + '/'
    save_dir = str(Path(save_dir))
    yolo_label_dir = save_dir + '/'
    """ yolo_image_dir = save_dir + 'images/'
    if not os.path.exists(yolo_image_dir):
        os.makedirs(yolo_image_dir) """
    if not os.path.exists(yolo_label_dir):
        os.makedirs(yolo_label_dir)
 
    json_files = glob(labelme_label_dir + '*.json')
    for ijf, jf in enumerate(json_files):
        print(ijf+1, '/', len(json_files), jf)
        filename = os.path.basename(jf).rsplit('.', 1)[0]
        labels, image_path = labelme2yolo_single(img_path,jf)
        if len(labels) > 0:
            np.savetxt(yolo_label_dir + filename + '.txt', labels)
            # shutil.copy(labelme_label_dir + image_path, yolo_image_dir + image_path)
    print('Completed!')

def trans1(input_dir, output_dir, word, splitword):
        for root, dirs, files in os.walk(input_dir):
           for item in files:
               if os.path.splitext(item)[1] == ".txt":
                   f = open(input_dir+item, "r", encoding='UTF-8')
                   content = f.read()
                   content = content.replace(word, splitword)
                   with open(os.path.join(output_dir, item), 'w', encoding='UTF-8') as fval:
                           fval.write(content)
                   f.close()
def ConvertELogStrToValue(eLogStr):
    """
    convert string of natural logarithm base of E to value
    return (convertOK, convertedValue)
    eg:
    input:  -1.1694737e-03
    output: -0.001169
    input:  8.9455025e-04
    output: 0.000895
    """
 
    (convertOK, convertedValue) = (False, 0.0)
    foundEPower = re.search("(?P<coefficientPart>-?\d+\.\d+)e(?P<ePowerPart>-\d+)", eLogStr, re.I)
    #print "foundEPower=",foundEPower
    if(foundEPower):
        coefficientPart = foundEPower.group("coefficientPart")
        ePowerPart = foundEPower.group("ePowerPart")
        #print "coefficientPart=%s,ePower=%s"%(coefficientPart, ePower)
        coefficientValue = float(coefficientPart)
        ePowerValue = float(ePowerPart)
        #print "coefficientValue=%f,ePowerValue=%f"%(coefficientValue, ePowerValue)
        #math.e= 2.71828182846
        # wholeOrigValue = coefficientValue * math.pow(math.e, ePowerValue)
        wholeOrigValue = coefficientValue * math.pow(10, ePowerValue)
 
        #print "wholeOrigValue=",wholeOrigValue;
 
        (convertOK, convertedValue) = (True, wholeOrigValue)
    else:
        (convertOK, convertedValue) = (False, 0.0)
 
    return (convertOK, convertedValue)
 
def parseIntEValue(intEValuesStr):
    # print "intEValuesStr=", intEValuesStr
    intEStrList = re.findall("-?\d+\.\d+e-\d+", intEValuesStr)
    # intEStrList = intEValuesStr.split(' ')
    # print "intEStrList=", intEStrList
    for eachIntEStr in intEStrList:
        # intValue = int(eachIntEStr)
        # print "intValue=",intValue
        (convertOK, convertedValue) = ConvertELogStrToValue(eachIntEStr)
        #print "convertOK=%s,convertedValue=%f"%(convertOK, convertedValue)
        print("eachIntEStr=%s,\tconvertedValue=%f" % (eachIntEStr, convertedValue))
        trans2(txt_path,txt_path,eachIntEStr,convertedValue)

def trans2(input_dir, output_dir, word, splitword):
         for root, dirs, files in os.walk(input_dir):
            for item in files:
                if os.path.splitext(item)[1] == ".txt":
                    f = open(input_dir+item, "r", encoding='UTF-8')
                    content = f.read()
                    content = content.replace(str(word), str(splitword))
                    with open(os.path.join(output_dir, item), 'w', encoding='UTF-8') as fval:
                            fval.write(content)
                    f.close()

if __name__ == '__main__':
    img_path = './images/'    # 数据集图片的路径
    json_dir = './json/'    # json标签的路径
    save_dir = './txt/'     # 保存的txt标签的路径
    labelme2yolo(img_path,json_dir, save_dir)
    input_dir = "./txt/"
    output_dir = "./txt/"
   # 要删除的字符
    word='+'
   # 要替换成的字符
    splitword = "-"
    trans1(input_dir, output_dir, word, splitword)
    txt_path = "txt/"
    output_dir = "txt/"
    # data_path = "D:/DeskTop/000001.txt"
    for root, dirs, files in os.walk(txt_path):
        for item in files:
            if os.path.splitext(item)[1] == ".txt":
                with open(txt_path + item, 'r') as f:
                    for line in f.readlines():
                        linestr = line.strip()
                        # print linestr
                        parseIntEValue(linestr)

