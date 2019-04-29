from pycocotools.coco import COCO
import skimage.io as io
import cv2
import json
import os
from multiprocessing import Pool
from functools import partial
import signal
import time
from PIL import Image, ImageFilter,ImageEnhance,ImageOps,ImageFile

def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])

def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def image_generation(category_name):

    print 'Category_name is %s'%(category_name)
    for dirpath,dirnames,filenames in os.walk(os.path.join(INSTANCE_FILE,category_name)):
        file_names = filenames
    for i in xrange(len(file_names)-1):
        file_name = file_names[i].split('.')[0]+'.pbm'
        ref_id = int(file_names[i].split('.')[0])
        ref_image_id = int(ann2img[ref_id])
        ref_mask = Image.open(os.path.join(INSTANCE_FILE,category_name,file_name))
        ref_x,ref_y,ref_w,ref_h = annotations[ref_id]['bbox']
        ref_xmin = int(round(ref_x))
        ref_ymin = int(round(ref_y))
        ref_xmax = int(round(ref_x + ref_w))
        ref_ymax = int(round(ref_y + ref_h))
        ref_instance_size = (ref_xmax - ref_xmin) * (ref_ymax - ref_ymin)
        if ref_xmax == ref_xmin or ref_ymax == ref_ymin : continue
        ref_mask = ref_mask.crop((ref_xmin,ref_ymin,ref_xmax,ref_ymax))
        ref_mask_array = 255-PIL2array1C(ref_mask)
        ref_mask_area = np.sum(ref_mask_array==255)

        for j in xrange(len(file_names)-1):
            pro_id = int(file_names[j].split('.')[0])
            pro_image_id = int(ann2img[pro_id])
            if ref_image_id==pro_image_id:continue
            file_name = file_names[j].split('.')[0]+'.pbm'
            mask = Image.open(os.path.join(INSTANCE_FILE,category_name,file_name))
            x,y,w,h = annotations[pro_id]['bbox']
            xmin = int(round(x))
            ymin = int(round(y))
            xmax = int(round(x + w))
            ymax = int(round(y + h))
            instance_size = (xmax - xmin) * (ymax - ymin)
            if instance_size != 0:
                ref2pro_ratio = float(ref_instance_size) / float(instance_size)
            else: ref2pro_ratio=0
            if xmax == xmin or ymax == ymin : continue
            mask = mask.crop((xmin,ymin,xmax,ymax))
            mask = mask.resize(((ref_xmax-ref_xmin),(ref_ymax-ref_ymin)), Image.ANTIALIAS)
            mask_array = 255-PIL2array1C(mask)
            mask_area = np.sum(mask_array==255)
            ssd = np.sum((ref_mask_array-mask_array)**2)
            if ref_mask_area != 0: ref_ssd2instance_ratio = float(ssd) / float(ref_mask_area)
            else: ref_ssd2instance_ratio = 1
            if mask_area != 0: pro_ssd2instance_ratio = float(ssd) / float(mask_area)
            else: pro_ssd2instance_ratio = 1
            if ref_ssd2instance_ratio > 0.3 or pro_ssd2instance_ratio > 0.3 : continue
            if ref2pro_ratio > 3 or ref2pro_ratio < 0.3: continue
            if ssd == 0 : continue
            ann2ann={}
            ann2ann[ref_id]=pro_id
            if not os.path.exists(os.path.join(ANNOTATION_FILE,category_name)):
                os.makedirs(os.path.join(ANNOTATION_FILE,category_name))
            f=open(os.path.join(ANNOTATION_FILE,category_name,'ann2ann.txt'),'a')
            f.write(str(ann2ann)+'\n')
            f.close()
            print 'Finding satisfactory Annotation_a %s and Annotation_b %s for class %s'%(ref_id,pro_id,category_name)
        print 'Finding annotation done for class'%(category_name)

def multi_process():
    partial_func = partial(image_generation)
    p = Pool(80,init_worker)
    try:
        p.map(partial_func, category_list)
    except KeyboardInterrupt:
        print "....\nCaught KeyboardInterrupt, terminating workers"
        p.terminate()
    else:
        p.close()
    p.join()
    print 'Mask Projection Done'


INVERTED_MASK = 1
INSTANCE_FILE = #MSCOCO instance mask output file path, e.g.'MSCOCO/masks/'
ANNOTATION_FILE = #annotation pair file path for each category,e.g.'MSCOCO/PSIS/'
JSON_FILE= #MSCOCO anntation json file path,e.g.'MSCOCO/annotations/instances_train2017.json'
os.makedirs(ANNOTATION_FILE)
fjson_src = open(JSON_FILE, 'r')
annotations = {annotation['id']:annotation for annotation in dataset_src['annotations']}

category_list = ['person', 'bicycle', 'car','motorcycle','airplane','bus','train', 'truck', 'boat','traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench','bird','cat','dog','horse','sheep',
                 'cow', 'elephant','bear','zebra','giraffe','backpack','umbrella', 'handbag','tie','suitcase',
                 'frisbee','skis','snowboard','sports ball','kite', 'baseball bat','baseball glove','skateboard','surfboard','tennis racket',
                 'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
                 'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
                 'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
                 'toaster', 'sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
multi_process()

print'Master: I\'m Done'
