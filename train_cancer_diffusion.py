from dataclasses import dataclass
import torch.nn.functional as F
import torch
from PIL import Image

from utils import *

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 1000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'output/gastro_images_mask_more_data_with_blur_100'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    
    with_mask = True
    
    num_train_timesteps = 100
    
    def __str__(self) -> str:
        pass

config = TrainingConfig()

from datasets import load_dataset
from dataset import SubCompose,GastroCancerDataset,get_test_samples

from torchvision import transforms

preprocess = SubCompose(    #transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


# config.dataset = "huggan/smithsonian_butterflies_subset"
# dataset = load_dataset(config.dataset, split="train")

# # Feel free to try other datasets from https://hf.co/huggan/ too! 
# # Here's is a dataset of flower photos:
# # config.dataset = "huggan/flowers-102-categories"
# # dataset = load_dataset(config.dataset, split="train")

# Or just load images from a local folder!
#config.dataset = "imagefolder"
#dataset = load_dataset(config.dataset, data_dir="nbi_v4/train_diff/3", split='train')

# dataset = GastroCancerDataset("gastro_cancer/xiehe_far_1",
#                                      "crop_images",
#                                      "annotations/crop_instances_default.json",
#                                      transforms = preprocess)

dataset_records = {"gastro_cancer/xiehe_far_1":[1], #0
                "gastro_cancer/xiehe_far_2":[1], #1
                "gastro_cancer/xiangya_far_2021":[1], #2
                "gastro_cancer/xiangya_far_2022":[1], #3
                "gastro_cancer/gastro8-12/2021-2022年癌变已标注/20221111/2021_2022_癌变_20221111":[1,4,5], #4
                "gastro_cancer/gastro8-12/低级别_2021_2022已标注/2021_2022_低级别_20221110":[1,4,5], #5
                "gastro_cancer/gastro8-12/协和2022_第一批胃早癌视频裁图已标注/20221115/癌变2022_20221115":[1,4,5], #6
                "gastro_cancer/gastro8-12/协和2022_第二批胃早癌视频裁图已标注/协和_2022_癌变_2_20221117":[1,4,5], #7
                "gastro_cancer/gastro8-12/协和21-11月~2022-5癌变已标注/协和2021-11月_2022-5癌变_20221121":[1,4,5], #8
                }
#dataset = GastroCancerDataset(root_dirs[0],
#                                transforms = preprocess)
dataset = None
for root_dir in dataset_records:
    if dataset ==None:
        dataset = GastroCancerDataset(root_dir,cat_ids=dataset_records[root_dir],
                                        transforms = preprocess) 
    else:       
        dataset += GastroCancerDataset(root_dir,cat_ids=dataset_records[root_dir],
                                        transforms = preprocess)

'''
def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
'''

import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

from diffusers import UNet2DModel


model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
      ),
)

from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

from diffusers import DDPMPipeline,DDPMPipelineMask

import math

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    if config.with_mask:
        bg_image,mask = get_test_samples(preprocess)
    else:
        bg_image,mask = None,None #get_test_samples(preprocess)
    
    images = pipeline( #该类的代码中，有一行代码image = (image / 2 + 0.5).clamp(0, 1)为denormalize
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
        bg_image = bg_image,
        mask = mask,
        num_inference_steps = config.num_train_timesteps,
        return_dict=False,
    )[0]
    #print('images: ', images)
    #print('images info: ', len(images), len(images[0]))

    # Make a grid out of the images
    image_grid = make_grid(images, rows=2, cols=2)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


from accelerate import Accelerator
# from diffusers.hub_utils import init_git_repo, push_to_hub

from tqdm.auto import tqdm
import os

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        # if config.push_to_hub:
        #    repo = init_git_repo(config, at_init=True)
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['image']
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            
            #todo:这里生成masks用于测试add_noise_with_mask
            #masks = generate_test_masks(clean_images.shape,clean_images.device)
            masks = batch['mask']

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            if config.with_mask:
                noisy_images = noise_scheduler.add_noise_with_mask(clean_images, masks, noise, timesteps)
            else:
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                masks = 1
                
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                temp_pred = model(noisy_images, timesteps)
                noise_pred = temp_pred["sample"]
                #loss = F.mse_loss(noise_pred, noise)
                #这里在计算noise的loss的时候需要添加掩码masks
                loss = F.mse_loss(noise_pred,noise*masks)
                
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipelineMask(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    push_to_hub(config, pipeline, repo, commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir) 

if __name__ == '__main__':
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
