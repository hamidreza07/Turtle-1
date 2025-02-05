import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
from PIL import Image
import os
import glob
import sys
from pathlib import Path
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")
from torch.cuda.amp import autocast

sys.path.append(str(Path(__file__).parents[1]))
from utils import tensor2img
sys.path.append(str(Path(__file__).parents[3]))
from basicsr.utils.options import parse
from importlib import import_module
placeholder_dp = "noise"

class Denoising(torch.utils.data.Dataset):
    def __init__(self, data_path, video,
                 noise_std,  
                 sample=True):
        # sample: if True, new data is created (since noise is random).
        super().__init__()
        self.data_path = data_path

        self.files =  sorted(glob.glob(data_path + "/*.*"))
        self.len = self.bound = len(self.files)
        self.current_frame = 0
        print(f"> # of Frames in {video}: {len(self.files)}")
        self.transform = transforms.Compose([transforms.ToTensor()])

        Img = Image.open(self.files[0])
        Img = np.array(Img)
        H, W, C = Img.shape

        os.makedirs(os.path.join(f"{placeholder_dp}/{video}_{int((noise_std)*255)}"), exist_ok=True)
        self.noisy_folder = os.path.join(f"{placeholder_dp}/{video}_{int((noise_std)*255)}")

        if sample:
            for i in range(self.len):
                Img = Image.open(self.files[i])
                Img = self.transform(Img)
                self.C, self.H, self.W = Img.shape
                std1 = noise_std
                noise = torch.empty_like(Img).normal_(mean=0, std=std1) #.cuda().half()
                Img = Img + noise
                
                np.save(os.path.join(self.noisy_folder, os.path.basename(self.files[i])[:-3]+"npy"), Img)
        
        self.noisy_files = sorted(glob.glob(os.path.join(self.noisy_folder, "*.npy")))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        Img = Image.open(self.files[index])
        noisy_Img = np.load(self.noisy_files[index])


        img_gt = np.array(Img)
        img_gt = self.transform(img_gt)
        img_in = torch.from_numpy(noisy_Img.copy())
        file_name = os.path.basename(self.noisy_files[index])
        return (file_name,  img_in.type(torch.FloatTensor))


class VideoLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, video):
        super().__init__()
        self.data_path = data_path
        self.in_files = sorted(glob.glob(data_path + "/*.*"))
        self.len = len(self.in_files)
        print(f"> # of Frames in {video}: {len(self.in_files)}")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])
        
        Img = Image.open(self.in_files[0])
        Img = np.array(Img)
        H, W, C = Img.shape
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_in = Image.open(self.in_files[index])
        img_in = np.array(img_in)
        file_name = os.path.basename(self.in_files[index])  # Extract the file name
        return (file_name, self.transform(np.array(img_in)).float())


def run_inference_full_image(prev_frame, curr_frame, model, device, chunk_size=1200, model_type="t1"):
    """
    Processes the image in vertical chunks to avoid memory issues and handles SR specifically.
    """
    # Handle SR-specific downscaling
    if model_type == "SR":
        prev_frame = F.interpolate(prev_frame.unsqueeze(0), scale_factor=0.25, mode="bicubic").squeeze(0)
        curr_frame = F.interpolate(curr_frame.unsqueeze(0), scale_factor=0.25, mode="bicubic").squeeze(0)

    # Ensure frames are 4D: (batch_size=1, channels, height, width)
    if prev_frame.dim() == 3:
        prev_frame = prev_frame.unsqueeze(0)
    if curr_frame.dim() == 3:
        curr_frame = curr_frame.unsqueeze(0)

    # Concatenate frames along the channel dimension
    x = torch.cat((prev_frame, curr_frame), dim=1).to(device)  # Shape: (1, 2C, H, W)

    # Reshape into (B, T=2, C, H, W)
    T = 2
    C = x.shape[1] // T
    x_reshaped = x.view(x.shape[0], T, C, x.shape[2], x.shape[3])

    b, t, c, h, w = x_reshaped.shape
    output = torch.zeros(b, c, h * (4 if model_type == "SR" else 1), w * (4 if model_type == "SR" else 1), device=device)

    # Process in vertical chunks
    for start in range(0, h, chunk_size):
        end = min(start + chunk_size, h)
        chunk = x_reshaped[:, :, :, start:end, :]

        with torch.no_grad(), autocast():
            out_chunk, _, _ = model(chunk.float(), None, None)

        if model_type == "SR":
            output[:, :, start * 4:end * 4, :] = torch.clamp(out_chunk, 0, 1)
        else:
            output[:, :, start:end, :] = torch.clamp(out_chunk, 0, 1)

    return output.squeeze(0)  # Remove batch dimension




def load_model(path, model):
    device = "cuda"
    model.load_state_dict(torch.load(path)['params'])
    model = model.to(device)
    model.eval()
    print(f"> Loaded Model.")
    return model, device


def run_inference(test_loader, model, device, model_type, save_img, image_out_path, chunk_size):
    previous_frame = None

    # Ensure the base path for saving images is created
    base_path = image_out_path
    os.makedirs(base_path, exist_ok=True)

    for ix in range(len(test_loader.dataset)):
        file_name, current_frame = test_loader.dataset[ix]
        print(file_name)

        current_frame = current_frame.to(device)
        if previous_frame is None:
            previous_frame = current_frame.clone()

        # Run full-image inference in chunks (no tiling)
        out_frame = run_inference_full_image(previous_frame, current_frame, model, device, chunk_size=chunk_size, model_type=model_type)
        out_frame = out_frame[:, :current_frame.shape[1], :current_frame.shape[2]]  # Just ensure size matches original

        if save_img:
            # Save the output using the original file name
            out_np = out_frame.permute(1, 2, 0).detach().cpu().numpy() * 255
            out_np = out_np.astype(np.uint8)
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            filename_no_ext, _ = os.path.splitext(file_name)  
            file_name_pred = os.path.join(base_path, filename_no_ext + '.png')
            cv2.imwrite(file_name_pred, out_bgr)


        previous_frame = current_frame.clone()

    return None, None


def create_video_model(opt, model_type="t0"):
    if model_type == "t0":
        module = import_module('basicsr.models.archs.turtle_arch')
        model = module.make_model(opt)
    elif model_type == "t1":
        module = import_module('basicsr.models.archs.turtle_t1_arch')
        model = module.make_model(opt)
    elif model_type == "SR":
        module = import_module('basicsr.models.archs.turtlesuper_t1_arch')
        model = module.make_model(opt)
    else:
        print("Model type not defined")
        exit()
    return model


def main(model_path,
         data_dir,
         config_file,
         save_image,
         model_type,
         image_out_path,
         chunk_size=192 ,
         task_name = "deblure",
        noise_sigma=5.0/255.0,
         sample=True,
         y_channel_PSNR=False):

    print(f"model_type: {model_type}")
    print(f"Using chunk processing with chunk_size: {chunk_size}")

    opt = parse(config_file, is_train=True)
    model = create_video_model(opt, model_type)

    model, device = load_model(model_path, model)
    if task_name == "denoise":
        data = Denoising(data_dir, None,noise_sigma)
    else:
        data = VideoLoader(data_dir, None)
    test_loader = torch.utils.data.DataLoader(data,
                                                batch_size=1, 
                                                num_workers=1, 
                                                shuffle=False)
    _, _ = run_inference(                            test_loader,
                            model,
                            device,
                            model_type,
                            save_img=save_image,
                            image_out_path=image_out_path,
                            chunk_size=chunk_size)

    return 0, 0


if __name__ == "__main__":
    st = time.time()

    # Example configuration:
    config = "options/Turtle_Deblur_Gopro.yml"
    model_path = "experiments/supervideos73/models/net_g_70000.pth" 
    data_dir = "demo_73"
    image_out_path = "result/demo_73"
    model_type = "t1"
    save_image = True
    chunk_size = 2000  # Adjust based on VRAM usage

    _, _ = main(model_path=model_path,
                config_file=config,
                data_dir=data_dir,
                model_type=model_type,
                save_image=save_image,
                image_out_path=image_out_path, 
                chunk_size=chunk_size)

    end = time.time()
    print(f"Completed in {end-st}s")
