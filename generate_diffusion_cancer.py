from diffusers import DDPMPipeline
from PIL import Image
import os

image_pipe = DDPMPipeline.from_pretrained('cancer_diffusion_results')
image_pipe.to('cuda')
output_dir = 'output_cancer_images_512'

for i in range(300):
	images = image_pipe().images
	output_img_name = 'generated_cancer_' + str(i) + '.jpg'
	images[0].save(os.path.join(output_dir, output_img_name))
