from datasets import iterable_dataset
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import os
import random
import cv2
from utils import *

class GastroCancerDataset(Dataset):
    def __init__(self,root_dir,image_folder='images',
                 ann_file_dir="annotations/instances_default.json",
                 cat_ids = [1,],transform=None,shuffle=False):
        self.root_dir = root_dir
        self.image_folder = image_folder
        self.annotation_file_dir = ann_file_dir
        self.cat_ids = cat_ids
        self.transform = transform
        self.shuffle = shuffle
        
        self.coco = COCO(os.path.join(root_dir,ann_file_dir))
        self.load_with_coco_per_ann()
        #self.load_with_coco_per_img()
        
    def load_with_coco_per_ann(self):
        #以每个病变为单例进行输入
        self.instances = []
        
        for ann_key in self.coco.anns:
            ann = self.coco.anns[ann_key]
            instance = {}
            if len(ann["segmentation"])>0 and len(ann["segmentation"][0])>0 and (ann['category_id'] in self.cat_ids):#这里默认每个目标只有一个segmentation标注
                instance["polygon"] = ann["segmentation"][0]
                instance["bbox"] = ann["bbox"]
                instance["cat_id"] = ann["category_id"]
                
                img = self.coco.loadImgs([ann["image_id"]])[0]
                instance["img_dir"] = os.path.join(self.root_dir,self.image_folder,img['file_name'])
                instance["img_shape"] = [img['width'],img['height']]
                if os.path.exists(instance["img_dir"]):
                    self.instances.append(instance)
        if self.shuffle:
            random.shuffle(self.instances)
            
    def load_with_coco_per_img(self):
        #以每张图片作为单例进行输入
        pass     
    
    def __len__(self):
        return len(self.instances)    
    
    def load_sample(self,instance):
        sample = {}
        image = cv2.imread(instance["img_dir"])
        image = image[int(instance['bbox'][1]):int(instance['bbox'][1]+instance['bbox'][3]),
                      int(instance['bbox'][0]):int(instance['bbox'][0]+instance['bbox'][2])]
        mask = poly2mask(*polygon2vertex_coords(instance['polygon']),(instance['img_shape'][1],instance['img_shape'][0]))
        mask = mask[int(instance['bbox'][1]):int(instance['bbox'][1]+instance['bbox'][3]),
                    int(instance['bbox'][0]):int(instance['bbox'][0]+instance['bbox'][2])]
        
        sample["image"] = image
        sample["mask"] = mask
        
        return sample
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        
        sample = self.load_sample(instance)

        if self.transform:
            sample = self.transform(sample)

        return sample        
        
if __name__ == '__main__':
    gc_dataset = GastroCancerDataset("gastro_cancer/xiehe_far_1","crop_images","annotations/crop_instances_default.json")
    gc_dataset[0]