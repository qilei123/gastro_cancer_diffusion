from skimage import draw
import numpy as np
import torch
import cv2

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

if __name__ == '__main__':
    test_gen_masks()