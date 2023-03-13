from datasets import iterable_dataset
from pycocotools.coco import COCO
from torch.utils.data import Dataset,BatchSampler
import os
import random
import cv2
from utils import *
from torchvision import transforms
from torchvision.transforms import Compose
from PIL import Image

DEBUG=True

class SubCompose(Compose):
    def __getitem__(self, index):
        return self.transforms[index]
    def __len__(self):
        return len(self.transforms)

class GastroCancerDataset(Dataset):
    def __init__(self,root_dir,image_folder='images',
                 ann_file_dir="annotations/instances_default.json",
                 cat_ids = [1,],transforms=None,shuffle=False, 
                 with_crop = False,blur_mask = False,bbox_extend = 1):
        self.root_dir = root_dir
        self.image_folder = image_folder
        self.annotation_file_dir = ann_file_dir
        self.cat_ids = cat_ids
        self.transforms = transforms
        self.shuffle = shuffle
        self.with_crop = with_crop
        self.blur_mask = blur_mask
        assert bbox_extend>=1,"bbox_extend should larger than 1"
        self.bbox_extend = bbox_extend-1
        
        self.instances = []
        
        self.coco = COCO(os.path.join(root_dir,ann_file_dir))
        self.load_with_coco_per_ann()
        #self.load_with_coco_per_img()
        
    def load_with_coco_per_ann(self):
        #以每个病变为单例进行输入
        for ann_key in self.coco.anns:
            ann = self.coco.anns[ann_key]
            instance = {}
            if len(ann["segmentation"])>0 and len(ann["segmentation"][0])>0 and (ann['category_id'] in self.cat_ids):#这里默认每个目标只有一个segmentation标注
                instance["polygon"] = ann["segmentation"][0]
                instance["bbox"] = ann["bbox"]
                
                if self.bbox_extend>0:
                    instance["bbox"][0] -= ann["bbox"][2]*self.bbox_extend/2
                    instance["bbox"][1] -= ann["bbox"][3]*self.bbox_extend/2
                    instance["bbox"][2] += ann["bbox"][2]*self.bbox_extend
                    instance["bbox"][3] += ann["bbox"][3]*self.bbox_extend
                    
                instance["cat_id"] = ann["category_id"]
                
                img = self.coco.loadImgs([ann["image_id"]])[0]
                instance["img_dir"] = os.path.join(self.root_dir,self.image_folder,img['file_name'])
                instance["img_shape"] = [img['width'],img['height']]
                
                #check the boundary of the bbox
                instance["bbox"][0] = 1 if instance["bbox"][0]<0 else instance["bbox"][0]
                instance["bbox"][1] = 1 if instance["bbox"][1]<0 else instance["bbox"][1]
                instance["bbox"][2] = img["width"]-instance["bbox"][0] if instance["bbox"][0]+instance["bbox"][2]>img["width"] else instance["bbox"][2]
                instance["bbox"][3] = img["height"]-instance["bbox"][1] if instance["bbox"][1]+instance["bbox"][3]>img["height"] else instance["bbox"][3]
                
                if os.path.exists(instance["img_dir"]):
                    self.instances.append(instance)
        if self.shuffle:
            random.shuffle(self.instances)
            
    def load_with_coco_per_img(self):
        #以每张图片作为单例进行输入
        pass     
    
    def __len__(self):
        return len(self.instances)    
    
    def __add__(self, other):
        self.instances  = self.instances + other.instances
        return self
    
    def load_sample(self,instance,index):
        sample = {}
        #opencv的加载方式来加载图片，需要注意通道的对应关系
        #image = cv2.imread(instance["img_dir"])
        #image = image[int(instance['bbox'][1]):int(instance['bbox'][1]+instance['bbox'][3]),
        #              int(instance['bbox'][0]):int(instance['bbox'][0]+instance['bbox'][2])]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(image)
        
        #采用PIL的方式来加载图片
        image = Image.open(instance["img_dir"])
        if self.with_crop:
            image = image.crop((int(instance['bbox'][0]),int(instance['bbox'][1]),
                                int(instance['bbox'][0]+instance['bbox'][2]),int(instance['bbox'][1]+instance['bbox'][3])))
        if DEBUG:
            image.save("temp_files/"+str(index)+".jpg")
        #print(instance['polygon'])
        mask = poly2mask(*polygon2vertex_coords(instance['polygon']),(instance['img_shape'][1],instance['img_shape'][0]))
        
        #todo:获得一个数最近的奇数,这里之后可以做kernal size
        #ksize = floor(instance['bbox'][2])+1-floor(instance['bbox'][2])%2
        
        if self.blur_mask:
            mask = cv2.GaussianBlur(mask, (15,15), 0)
        
        if self.with_crop:
            mask = mask[int(instance['bbox'][1]):int(instance['bbox'][1]+instance['bbox'][3]),
                        int(instance['bbox'][0]):int(instance['bbox'][0]+instance['bbox'][2])]
        
        #mask = Image.fromarray(mask)
        
        #mask = mask.crop((int(instance['bbox'][0]),int(instance['bbox'][1]),
        #                    int(instance['bbox'][0]+instance['bbox'][2]),int(instance['bbox'][1]+instance['bbox'][3])))        
        if DEBUG:
            cv2.imwrite("temp_files/"+str(index)+".png",mask*255)
        #mask.save("temp_files/"+str(index)+".jpg")
        
        sample["image"] = image
        sample["mask"] = mask
        
        return sample
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        
        sample = self.load_sample(instance,index=idx)

        if self.transforms:
            image_tensor = self.transforms[0](sample['image']) #totensor
            mask_tensor = self.transforms[0](sample['mask']) #totensor
            #mask_tensor = torch.stack((mask_tensor,mask_tensor,mask_tensor),dim=1)
            mask_tensor = mask_tensor.repeat(3,1,1)
            
            image_mask_stack = torch.cat((image_tensor,mask_tensor)).to(torch.float32) 
            
            for i in range(1,len(self.transforms)-1):
                image_mask_stack = self.transforms[i](image_mask_stack)
            #sample['image'] = self.transforms[:-1](sample['image'])
            sample['image'],sample['mask'] = torch.split(image_mask_stack, 3)
            sample['image'] = self.transforms[-1](sample["image"])
        return sample        

def get_test_samples(preprocess,with_crop=False,blur_mask = False,bbox_extend=1):
    #todo:这里需要将四张测试图片加载进来，并且需要进行preprocess
    test_images_record = open('choose_test_gastro_images.txt')
    
    records = []
    
    line = test_images_record.readline()
    
    bbox_extend-=1
    
    while line:
        
        records.append(line[:-1])
        
        line = test_images_record.readline()
        
    image_list = records[::2]
    ann_list = records[1::2]
    
    sample_images = []
    sample_masks = []
    
    for image_name,ann in zip(image_list,ann_list):
        ann = json.loads(ann)
        
        if bbox_extend>0:
            ann["bbox"][0] -= ann["bbox"][2]*bbox_extend/2
            ann["bbox"][1] -= ann["bbox"][3]*bbox_extend/2
            ann["bbox"][2] += ann["bbox"][2]*bbox_extend
            ann["bbox"][3] += ann["bbox"][3]*bbox_extend
        
        sample = {}
        
        image = Image.open(os.path.join("gastro_images_test",image_name))
        
        height = image.height
        
        width = image.width
        
        ann["bbox"][0] = 1 if ann["bbox"][0]<0 else ann["bbox"][0]
        ann["bbox"][1] = 1 if ann["bbox"][1]<0 else ann["bbox"][1]
        ann["bbox"][2] = width-ann["bbox"][0] if ann["bbox"][0]+ann["bbox"][2]>width else ann["bbox"][2]
        ann["bbox"][3] = height-ann["bbox"][1] if ann["bbox"][1]+ann["bbox"][3]>height else ann["bbox"][3]        
        
        if with_crop:
            image = image.crop((int(ann['bbox'][0]),int(ann['bbox'][1]),
                        int(ann['bbox'][0]+ann['bbox'][2]),int(ann['bbox'][1]+ann['bbox'][3])))
        
        sample["image"] = image
        if DEBUG:
            image.save("temp_files/"+image_name)

        mask = poly2mask(*polygon2vertex_coords(ann["segmentation"][0]),(height,width))
        
        if blur_mask:
            mask = cv2.GaussianBlur(mask, (15,15), 0)
        
        if with_crop:
            mask = mask[int(ann['bbox'][1]):int(ann['bbox'][1]+ann['bbox'][3]),
                        int(ann['bbox'][0]):int(ann['bbox'][0]+ann['bbox'][2])]   
        sample["mask"]= mask
        if DEBUG:
            cv2.imwrite("temp_files/"+image_name+".png",mask*255)     
        if preprocess:
            image_tensor = preprocess[0](sample['image']) #totensor
            mask_tensor = preprocess[0](sample['mask']) #totensor
            #mask_tensor = torch.stack((mask_tensor,mask_tensor,mask_tensor),dim=1)
            mask_tensor = mask_tensor.repeat(3,1,1)
            
            image_mask_stack = torch.cat((image_tensor,mask_tensor)).to(torch.float32) 
            
            #for i in range(1,len(preprocess)-1): #不需要再做flip
            image_mask_stack = preprocess[1](image_mask_stack)#scale操作
            #sample['image'] = self.transforms[:-1](sample['image'])
            sample['image'],sample['mask'] = torch.split(image_mask_stack, 3)
            sample['image'] = preprocess[4](sample["image"])  #只在image上做normalize 
            
        sample_images.append(sample['image'])
        sample_masks.append(sample["mask"])
        
    images = torch.stack(sample_images)
    masks = torch.stack(sample_masks)
    
    #mask_images = images*masks
        
    return images,masks  


def test_dataset():
    preprocess = SubCompose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    root_dirs = ["gastro_cancer/xiehe_far_1",
                 "gastro_cancer/xiehe_far_2",
                 "gastro_cancer/xiangya_far_2021",
                 "gastro_cancer/xiangya_far_2022"]
    gc_dataset = GastroCancerDataset(root_dirs[0],
                                    "crop_images",
                                    "annotations/crop_instances_default.json",
                                    transforms = preprocess,with_crop=True,blur_mask=True)
    
    for root_dir in root_dirs[1:]:
        print(root_dir)
        gc_dataset += GastroCancerDataset(root_dir,
                                        "crop_images",
                                        "annotations/crop_instances_default.json",
                                        transforms = preprocess,blur_mask=True)
    #gc_dataset[324]
    for i in range(len(gc_dataset)):
        gc_dataset[i]    

def test_get_test_samples():
    preprocess = SubCompose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    get_test_samples(preprocess,with_crop=True)

if __name__ == '__main__':
    test_dataset()
    #test_get_test_samples()