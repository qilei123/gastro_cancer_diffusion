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
from math import floor
import pickle
from tqdm import tqdm
from PIL import ImageFilter
from scipy import ndimage


DEBUG=False

dataset_names = ['dataset1','dataset2']

dataset_records = {"dataset1":
                {"gastro_cancer/xiehe_far_1":[1], #0
                "gastro_cancer/xiehe_far_2":[1], #1
                "gastro_cancer/xiangya_far_2021":[1], #2
                "gastro_cancer/xiangya_far_2022":[1], #3
                },
                "dataset2":
                {"gastro_cancer/xiehe_far_1":[1], #0
                "gastro_cancer/xiehe_far_2":[1], #1
                "gastro_cancer/xiangya_far_2021":[1], #2
                "gastro_cancer/xiangya_far_2022":[1], #3
                "gastro_cancer/gastro8-12/2021-2022年癌变已标注/20221111/2021_2022_癌变_20221111":[1,4,5], #4
                "gastro_cancer/gastro8-12/低级别_2021_2022已标注/2021_2022_低级别_20221110":[1,4,5], #5
                "gastro_cancer/gastro8-12/协和2022_第一批胃早癌视频裁图已标注/20221115/癌变2022_20221115":[1,4,5], #6
                "gastro_cancer/gastro8-12/协和2022_第二批胃早癌视频裁图已标注/协和_2022_癌变_2_20221117":[1,4,5], #7
                "gastro_cancer/gastro8-12/协和21-11月~2022-5癌变已标注/协和2021-11月_2022-5癌变_20221121":[1,4,5], #8
                }
                }

class SubCompose(Compose):
    def __getitem__(self, index):
        return self.transforms[index]
    def __len__(self):
        return len(self.transforms)

class GastroCancerDataset(Dataset):
    def __init__(self,root_dir,dataset_name = 'dataset1',image_folder='images',
                 ann_file_dir="annotations/instances_default.json",
                 cat_ids = [1,],transforms=None,shuffle=False, 
                 with_crop = False,
                 blur_mask = False, dynamic_blur_mask = False, blur_kernel_scale = 10,
                 bbox_extend = 1):
        if dataset_name is not None:
            assert dataset_name in dataset_records,"No dataset name: "+dataset_name  
            
        self.dataset_name = dataset_name  
    
        self.root_dir = root_dir
        self.image_folder = image_folder
        self.annotation_file_dir = ann_file_dir
        self.cat_ids = cat_ids
        self.transforms = transforms
        self.shuffle = shuffle
        self.with_crop = with_crop
        self.blur_mask = blur_mask
        self.dynamic_blur_mask = dynamic_blur_mask
        self.blur_kernel_scale = blur_kernel_scale
        assert bbox_extend>=1,"bbox_extend should larger than 1"
        self.bbox_extend = bbox_extend
        
        self.instances = []
        self.cache_datas = []
        
        '''
        #todo:待开发，采用大文件方式存储数据
        if dataset_name is None:
            self.cache_datas = []
        elif not os.path.isfile(dataset_name+".cache"):
            self.cache_datas = []
        else:
            fptr = open(dataset_name+".cache", "rb")
            self.cache_datas = pickle.load(fptr)
            fptr.close()
        '''
        
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
                    
                instance["cat_id"] = ann["category_id"]
                
                img = self.coco.loadImgs([ann["image_id"]])[0]
                instance["img_dir"] = os.path.join(self.root_dir,self.image_folder,img['file_name'])
                instance["img_shape"] = [img['width'],img['height']]
                instance["img_width"],instance["img_height"] = img['width'],img['height']
                
                #check the boundary of the bbox
                instance["bbox"][0] = 1 if instance["bbox"][0]<0 else instance["bbox"][0]
                instance["bbox"][1] = 1 if instance["bbox"][1]<0 else instance["bbox"][1]
                instance["bbox"][2] = img["width"]-instance["bbox"][0] if instance["bbox"][0]+instance["bbox"][2]>img["width"] else instance["bbox"][2]
                instance["bbox"][3] = img["height"]-instance["bbox"][1] if instance["bbox"][1]+instance["bbox"][3]>img["height"] else instance["bbox"][3]
                
                if os.path.exists(instance["img_dir"]):
                    self.instances.append(instance)
        #if self.shuffle:
        #    random.shuffle(self.instances)
            
    def load_with_coco_per_img(self):
        #以每张图片作为单例进行输入
        pass     
    
    def load_cache(self):
        #todo:待开发，采用大文件方式存储数据
        if self.dataset_name is None:
            self.cache_datas = []
        elif not os.path.isfile(self.dataset_name+".cache"):
            self.cache_datas = []
        else:
            fptr = open(self.dataset_name+".cache", "rb")
            self.cache_datas = pickle.load(fptr)
            fptr.close() 
            return True
        return False
            
    def save_cache(self):
        if self.dataset_name is not  None:
            if not os.path.isfile(self.dataset_name+".cache"):
                fptr = open(self.dataset_name+".cache", "wb")  # open file in write binary mode
                pickle.dump(self.cache_datas, fptr)  # dump list data into file 
                fptr.close()                
    
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
        if len(self.instances) > len(self.cache_datas):
            #采用PIL的方式来加载图片
            image = Image.open(instance["img_dir"])

            if self.bbox_extend>1:
                instance["bbox"][0] -= instance["bbox"][2]*(self.bbox_extend-1)/2
                instance["bbox"][1] -= instance["bbox"][3]*(self.bbox_extend-1)/2
                instance["bbox"][2] = instance["bbox"][2]*self.bbox_extend
                instance["bbox"][3] = instance["bbox"][3]*self.bbox_extend  
  
            #check the boundary of the bbox
            instance["bbox"][0] = 1 if instance["bbox"][0]<0 else instance["bbox"][0]
            instance["bbox"][1] = 1 if instance["bbox"][1]<0 else instance["bbox"][1]
            instance["bbox"][2] = instance["img_width"]-instance["bbox"][0] if instance["bbox"][0]+instance["bbox"][2]>instance["img_width"] else instance["bbox"][2]
            instance["bbox"][3] = instance["img_height"]-instance["bbox"][1] if instance["bbox"][1]+instance["bbox"][3]>instance["img_height"] else instance["bbox"][3]          
            
            if self.with_crop:
                image = image.crop((int(instance['bbox'][0]),int(instance['bbox'][1]),
                                    int(instance['bbox'][0]+instance['bbox'][2]),int(instance['bbox'][1]+instance['bbox'][3])))
            if DEBUG:
                image.save("temp_files/"+str(index)+".jpg")
            #print(instance['polygon'])
            mask = poly2mask(*polygon2vertex_coords(instance['polygon']),(instance['img_shape'][1],instance['img_shape'][0]))
            
            #todo:获得一个数最近的奇数,这里之后可以做kernal size
            if self.dynamic_blur_mask:
                min_box_edge = min(instance['bbox'][2],instance['bbox'][3])
                ksize = floor(min_box_edge/self.blur_kernel_scale)+1-floor(min_box_edge/self.blur_kernel_scale)%2
            else:
                ksize = 15
            
            if self.blur_mask:
                mask = cv2.GaussianBlur(mask, (ksize,ksize), 0)
            
            if self.with_crop:
                mask = mask[int(instance['bbox'][1]):int(instance['bbox'][1]+instance['bbox'][3]),
                            int(instance['bbox'][0]):int(instance['bbox'][0]+instance['bbox'][2])]
            
            #mask = Image.fromarray(mask)
            
            #mask = mask.crop((int(instance['bbox'][0]),int(instance['bbox'][1]),
            #                    int(instance['bbox'][0]+instance['bbox'][2]),int(instance['bbox'][1]+instance['bbox'][3])))        
            if DEBUG:
                cv2.imwrite("temp_files/"+str(index)+".png",(mask*0.5+0.2)*255)
            #mask.save("temp_files/"+str(index)+".jpg")
            
            sample["image"] = image
            sample["mask"] = mask
            self.cache_datas.append(sample) 
        elif len(self.instances) == len(self.cache_datas):
            sample = self.cache_datas[index]
            '''todo:cache待开发'''
            '''
            if self.dataset_name is not  None:
                if not os.path.isfile(self.dataset_name+".cache"):
                    fptr = open(self.dataset_name+".cache", "wb")  # open file in write binary mode
                    pickle.dump(self.cache_datas, fptr)  # dump list data into file 
                    fptr.close()    
            '''
        
        return sample
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        
        sample = self.load_sample(instance,index=idx)
        
        tensor_sample = {}

        if self.transforms:
            image_tensor = self.transforms[0](sample['image']) #totensor
            mask_tensor = self.transforms[0](sample['mask']) #totensor
            #mask_tensor = torch.stack((mask_tensor,mask_tensor,mask_tensor),dim=1)
            mask_tensor = mask_tensor.repeat(3,1,1)
            
            image_mask_stack = torch.cat((image_tensor,mask_tensor)).to(torch.float32) 
            
            for i in range(1,len(self.transforms)-1):
                image_mask_stack = self.transforms[i](image_mask_stack)
            #sample['image'] = self.transforms[:-1](sample['image'])
            tensor_sample['image'],tensor_sample['mask'] = torch.split(image_mask_stack, 3)
            tensor_sample['image'] = self.transforms[-1](tensor_sample["image"])
        return tensor_sample        

def get_test_samples(preprocess,with_crop=False,blur_mask = False,
                     dynamic_blur_mask = False,blur_kernel_scale = 10,
                     bbox_extend=1):
    #todo:这里需要将四张测试图片加载进来，并且需要进行preprocess
    test_images_record = open('choose_test_gastro_images.txt')
    
    records = []
    
    line = test_images_record.readline()
    
    #bbox_extend-=1
    
    while line:
        
        records.append(line[:-1])
        
        line = test_images_record.readline()
        
    image_list = records[::2]
    ann_list = records[1::2]
    
    sample_images = []
    sample_masks = []
    
    for image_name,ann in zip(image_list,ann_list):
        ann = json.loads(ann)
        
        if bbox_extend>1:
            ann["bbox"][0] -= ann["bbox"][2]*(bbox_extend-1)/2
            ann["bbox"][1] -= ann["bbox"][3]*(bbox_extend-1)/2
            ann["bbox"][2] = ann["bbox"][2]*bbox_extend
            ann["bbox"][3] = ann["bbox"][3]*bbox_extend
        
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
        
        #添加掩码边缘高斯模糊的操作
        if dynamic_blur_mask:
            min_box_edge = min(ann['bbox'][2],ann['bbox'][3])
            ksize = floor(min_box_edge/blur_kernel_scale)+1-floor(min_box_edge/blur_kernel_scale)%2
        else:
            ksize = 15
        
        if blur_mask:
            mask = cv2.GaussianBlur(mask, (ksize,ksize), 0)
        
        if with_crop:
            mask = mask[int(ann['bbox'][1]):int(ann['bbox'][1]+ann['bbox'][3]),
                        int(ann['bbox'][0]):int(ann['bbox'][0]+ann['bbox'][2])]   
        sample["mask"]= mask
        if DEBUG:
            cv2.imwrite("temp_files/"+image_name+".png",(mask*0.5+0.2)*255)     
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
    
    dataset_name='dataset1'
    dataset_record = dataset_records[dataset_name]
    #dataset = GastroCancerDataset(root_dirs[0],
    #                                transforms = preprocess)
    dataset = None
    for root_dir in dataset_record:
        if dataset ==None:
            dataset = GastroCancerDataset(root_dir,cat_ids=dataset_record[root_dir],dataset_name=dataset_name,
                                            transforms = preprocess,with_crop=True,
                                            blur_mask=False,
                                            dynamic_blur_mask=False,
                                            blur_kernel_scale=10,
                                            bbox_extend=1.5) 
            

        else:       
            dataset += GastroCancerDataset(root_dir,cat_ids=dataset_record[root_dir],dataset_name=dataset_name,
                                            transforms = preprocess)

    print("Loading dataset...")
    for i in tqdm(range(len(dataset))):
        dataset[i]  

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
    get_test_samples(preprocess,with_crop=True,blur_mask = False,dynamic_blur_mask=True,bbox_extend=1.5)

def test_patchs():
    test_images_record = open('choose_test_gastro_images.txt')
    
    #patchs_dir = "output/gastro_images_mask_data1_256_n600_crop_be1/samples/0279.png"
    #patchs_dir = "output/gastro_images_mask_data1_256_n500_crop_be1/samples/0439.png"
    
    patchs_dir = "output/gastro_images_mask_data2_256_nr800_crop_be1/samples/0149.png"
    
    patchs = Image.open(patchs_dir)
    
    patch_images = []
    patch_images.append(patchs.crop((0,0,patchs.width/2,patchs.height/2)))
    patch_images.append(patchs.crop((patchs.width/2,0,patchs.width,patchs.height/2)))
    patch_images.append(patchs.crop((0,patchs.height/2,patchs.width/2,patchs.height)))
    patch_images.append(patchs.crop((patchs.width/2,patchs.height/2,patchs.width,patchs.height)))
    
    records = []
    
    line = test_images_record.readline()
    
    #bbox_extend-=1
    
    while line:
        
        records.append(line[:-1])
        
        line = test_images_record.readline()
        
    image_list = records[::2]
    ann_list = records[1::2]
    
    
    for image_name,ann,patch in zip(image_list,ann_list,patch_images):
        
        ann = json.loads(ann)    
        image = Image.open(os.path.join("gastro_images_test",image_name))

        mask = poly2mask(*polygon2vertex_coords(ann["segmentation"][0]),(image.height,image.width))
        
        patch = patch.resize((int(ann['bbox'][2]),int(ann['bbox'][3])))
        patch_image = Image.new("RGB",image.size,(0,0,0))
        patch_image.paste(patch, (int(ann['bbox'][0]),int(ann['bbox'][1])))
        
        mask = ndimage.binary_erosion(mask,iterations=20).astype(mask.dtype)
        
        mask = Image.fromarray(mask*255/2).convert('L')
        mask = mask.filter(ImageFilter.GaussianBlur(10))
        image.paste(patch_image, (0,0),mask)
        
        image.save(os.path.join("results",image_name))

from diffusers import DiffusionPipeline
def test_diffusion_inpainting():
    
    pipe = DiffusionPipeline.from_pretrained(    
                        "runwayml/stable-diffusion-inpainting",
                        custom_pipeline="img2img_inpainting",
                        torch_dtype=torch.float16)
    
    pipe.safety_checker = lambda images, clip_input: (images, False)
    
    pipe = pipe.to('cuda:3')
    
    test_images_record = open('choose_test_gastro_images.txt')
    
    #patchs_dir = "output/gastro_images_mask_data1_256_n600_crop_be1/samples/0279.png"
    #patchs_dir = "output/gastro_images_mask_data1_256_n500_crop_be1/samples/0439.png"
    
    patchs_dir = "output/gastro_images_mask_data2_256_nr800_crop_be1/samples/0149.png"
    
    patchs = Image.open(patchs_dir)
    
    patch_images = []
    patch_images.append(patchs.crop((0,0,patchs.width/2,patchs.height/2)))
    patch_images.append(patchs.crop((patchs.width/2,0,patchs.width,patchs.height/2)))
    patch_images.append(patchs.crop((0,patchs.height/2,patchs.width/2,patchs.height)))
    patch_images.append(patchs.crop((patchs.width/2,patchs.height/2,patchs.width,patchs.height)))
    
    records = []
    
    line = test_images_record.readline()
    
    #bbox_extend-=1
    
    while line:
        
        records.append(line[:-1])
        
        line = test_images_record.readline()
        
    image_list = records[::2]
    ann_list = records[1::2]
    
    
    for image_name,ann,patch in zip(image_list,ann_list,patch_images):
        
        ann = json.loads(ann)    
        image = Image.open(os.path.join("gastro_images_test",image_name))

        mask = poly2mask(*polygon2vertex_coords(ann["segmentation"][0]),(image.height,image.width))
        
        patch = patch.resize((int(ann['bbox'][2]),int(ann['bbox'][3])))
        patch_image = Image.new("RGB",image.size,(0,0,0))
        patch_image.paste(patch, (int(ann['bbox'][0]),int(ann['bbox'][1])))
        
        #mask = ndimage.binary_erosion(mask,iterations=20).astype(mask.dtype)
        
        #mask = Image.fromarray(mask*255/2).convert('L')
        #mask = mask.filter(ImageFilter.GaussianBlur(10))
        #image.paste(patch_image, (0,0),mask)
        img_resize = (512,512)
        init_image = image.convert("RGB").resize(img_resize)
        inner_image = patch_image.convert("RGBA").resize(img_resize)
        mask_image = Image.fromarray(mask*255).convert("RGB").resize(img_resize)
        prompt = ""
        image = pipe(prompt=prompt, image=init_image, inner_image=inner_image, mask_image=mask_image).images[0]        
        image.save(os.path.join("results",image_name))

if __name__ == '__main__':
    #test_dataset()
    #test_get_test_samples()
    #test_patchs()
    test_diffusion_inpainting()
    pass