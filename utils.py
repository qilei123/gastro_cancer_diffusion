from skimage import draw
import numpy as np
import torch
import cv2
import json
import os

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def polygon2vertex_coords(polygon):
    col_coords = polygon[::2]
    row_coords = polygon[1::2]
    return [row_coords,col_coords]

def generate_masks(polygons, bs_shape, device):
    pass

def generate_test_masks(bs_shape,device):
    polygon_row_coords = [30, 90, 90, 30]
    polygon_col_coords = [30, 30, 90, 90]
    
    mask = poly2mask(polygon_row_coords,polygon_col_coords,(bs_shape[2],bs_shape[3]))
    
    mask_c3 = np.stack([mask]*bs_shape[1])
    
    masks = np.stack([mask_c3]*bs_shape[0])
    
    return torch.from_numpy(masks).to(device).to(torch.float32)
    

def test_gen_masks():
    polygon_row_coords = [30, 190, 90, 30]
    polygon_col_coords = [30, 300, 90, 90]
    
    mask = (poly2mask(polygon_row_coords,polygon_col_coords,(256,256))+0.5)*100
    
    mask_c3 = np.stack([mask]*3)
    
    cv2.imwrite('test_mask.jpg',mask)

def test_images():
    test_images_record = open('choose_test_gastro_images.txt')
    
    records = []
    
    line = test_images_record.readline()
    
    while line:
        
        records.append(line[:-1])
        
        line = test_images_record.readline()
        
    image_list = records[::2]
    ann_list = records[1::2]
    
    for image_name,ann in zip(image_list,ann_list):
        image = cv2.imread(os.path.join("gastro_images_test",image_name))

        #这里将原图像发大原来的1.2倍，以接近于训练集的图片尺寸，只需要进行一次操作即可
        '''
        scale_percent = 120 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        
        #cv2.imwrite(os.path.join("gastro_images_test",image_name),image) 
        '''
        ann = json.loads(ann)
        
        cv2.rectangle(image, (int(ann['bbox'][0]),int(ann['bbox'][1])),
                      (int(ann['bbox'][0]+ann['bbox'][2]),int(ann['bbox'][1]+ann['bbox'][3])),(255,0,0),2)
        
        cv2.imwrite(image_name,image)


if __name__ == '__main__':
    #test_gen_masks()
    test_images()