import os
import time
import sys
import argparse
import math
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from functools import partial
from torchvision import transforms
from PIL import Image
import wandb
import torchvision.utils as vutils
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.NAFNet_arch import NAFNet
from networks.shadow_matte import ShadowMattePredictor
from networks.Split_images import process_split_image_with_shadow_matte
from utils.UTILS1 import compute_psnr
from utils.UTILS import compute_ssim
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_image_overlap(img, crop_size, overlap_size):
    B, C, H, W = img.shape
    stride = crop_size - overlap_size
    y_starts = list(range(0, H - crop_size + 1, stride))
    if y_starts and y_starts[-1] != H - crop_size:
        y_starts.append(H - crop_size)
    x_starts = list(range(0, W - crop_size + 1, stride))
    if x_starts and x_starts[-1] != W - crop_size:
        x_starts.append(W - crop_size)
    patches = []
    positions = []
    for y in y_starts:
        for x in x_starts:
            patch = img[:, :, y:y+crop_size, x:x+crop_size]
            patches.append(patch)
            positions.append((y, x, crop_size, crop_size))
    return patches, positions

def create_gaussian_mask(patch_size, overlap):
    h, w = patch_size
    weight_y = torch.ones(h, dtype=torch.float32)
    sigma = overlap / 2.0 if overlap > 0 else 1.0
    for i in range(h):
        if i < overlap:
            weight_y[i] = math.exp(-0.5 * ((overlap - i)/sigma)**2)
        elif i > h - overlap - 1:
            weight_y[i] = math.exp(-0.5 * ((i - (h - overlap - 1))/sigma)**2)
    weight_x = torch.ones(w, dtype=torch.float32)
    for j in range(w):
        if j < overlap:
            weight_x[j] = math.exp(-0.5 * ((overlap - j)/sigma)**2)
        elif j > w - overlap - 1:
            weight_x[j] = math.exp(-0.5 * ((j - (w - overlap - 1))/sigma)**2)
    mask = torch.ger(weight_y, weight_x)
    return mask.unsqueeze(0).unsqueeze(0)

def merge_image_overlap(patches, positions, crop_size, resolution, overlap_size, blend_mode='gaussian'):
    B, C, H, W = resolution
    device = patches[0].device
    merged = torch.zeros((B, C, H, W), device=device)
    weight_sum = torch.zeros((B, 1, H, W), device=device)
    for patch, pos in zip(patches, positions):
        y, x, ph, pw = pos
        if blend_mode == 'gaussian' and overlap_size > 0:
            mask = create_gaussian_mask((ph, pw), overlap_size).to(device)
        else:
            mask = torch.ones((1, 1, ph, pw), device=device)
        merged[:, :, y:y+ph, x:x+pw] += patch * mask
        weight_sum[:, :, y:y+ph, x:x+pw] += mask
    merged = merged / (weight_sum + 1e-8)
    return merged

def sliding_crop_left(img, patch_size, stride):
    B, C, H, W = img.shape
    patches = []
    positions = []
    for x in range(0, W - patch_size + 1, stride):
        patch = img[:, :, 0:H, x:x+patch_size]
        patches.append(patch)
        positions.append((0, x, patch_size, H))
    return patches, positions

def merge_sliding_crops(crops, crop_positions, original_width, overlap_size):
    B, C, H, W_crop = crops[0].shape
    device = crops[0].device
    merged = torch.zeros((B, C, H, original_width), device=device)
    weight = torch.zeros((B, 1, H, original_width), device=device)
    for crop, pos in zip(crops, crop_positions):
        x = pos[1]
        merged[:, :, :, x:x+W_crop] += crop
        weight[:, :, :, x:x+W_crop] += 1.0
    merged = merged / (weight + 1e-8)
    return merged

class InferenceDataset(Dataset):
    def __init__(self, input_dir, gt_dir=None, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.image_names = sorted(os.listdir(input_dir))
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.has_gt = gt_dir is not None
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        img_path = os.path.join(self.input_dir, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.has_gt:
            gt_name = self.image_names[idx].replace('_in.', '_gt.')
            gt_path = os.path.join(self.gt_dir, gt_name)
            if not os.path.exists(gt_path):
                gt_path = os.path.join(self.gt_dir, self.image_names[idx])
            gt = Image.open(gt_path).convert('RGB')
            if self.transform:
                gt = self.transform(gt)
            return img, gt, self.image_names[idx]
        return img, self.image_names[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference: ViT+NAFNet pipeline with matte predictor and sliding crop")
    parser.add_argument("--checkpoint", type=str, default="./files/nafnet.pth")
    parser.add_argument("--vit_checkpoint", type=str, default="./files/vit.pth")
    parser.add_argument("--matte_generator_checkpoint", type=str, default="./files/shadow_matte_generator.pth")
    parser.add_argument("--input_dir", type=str, default="./test/")
    parser.add_argument("--gt_dir", type=str, default=None, help="GT images directory for evaluation")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nafn_patch_size", type=int, default=256)
    parser.add_argument("--overlap_size", type=int, default=128)
    parser.add_argument("--crop_stride", type=int, default=250)
    parser.add_argument("--img_size", type=int, default=None, help="Auto-detect if None")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    net = MaskedAutoencoderViT(
        patch_size=8, embed_dim=256, depth=6, num_heads=8,
        decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    net_1 = NAFNet(
        img_channel=3, width=32, middle_blk_num=24,
        enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], global_residual=False
    )
    net.to(device)
    net_1.to(device)
    net.eval()
    net_1.eval()
    
    if not os.path.exists(args.vit_checkpoint):
        print(f"[Error] ViT checkpoint not found: {args.vit_checkpoint}")
        sys.exit(1)
    net.load_state_dict(torch.load(args.vit_checkpoint, map_location=device))
    print(f"Loaded ViT checkpoint from {args.vit_checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        print(f"[Error] NAFNet checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    net_1.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded NAFNet checkpoint from {args.checkpoint}")
    
    dataset = InferenceDataset(args.input_dir, args.gt_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    count = 0
    
    s_time = time.time()
    with torch.no_grad():
        for data in loader:
            if args.gt_dir is not None:
                inputs, gts, fname = data
            else:
                inputs, fname = data
                gts = None
            inputs = inputs.to(device)
            if gts is not None:
                gts = gts.to(device)
            B, C, H, W = inputs.shape
            
            img_size = args.img_size if args.img_size is not None else H
            crop_stride = args.crop_stride if args.crop_stride is not None else W
            
            crops, crop_positions = sliding_crop_left(inputs, patch_size=img_size, stride=crop_stride)
            processed_crops = []
            processed_mattes = []
            
            for crop in crops:
                sub_data, positions = split_image_overlap(crop, crop_size=args.nafn_patch_size, overlap_size=args.overlap_size)
                matte_patches_input = []
                for sub in sub_data:
                    matte_patch = matte_predictor.predict_from_tensor(sub)
                    matte_patches_input.append(matte_patch)
                processed = process_split_image_with_shadow_matte(sub_data, matte_patches_input, net)
                matte_from_processed = []
                for proc_patch in processed:
                    matte_patch = matte_predictor.predict_from_tensor(proc_patch)
                    matte_from_processed.append(matte_patch)
                crop_processed = merge_image_overlap(processed, positions, crop_size=args.nafn_patch_size,
                                                     resolution=crop.shape, overlap_size=args.overlap_size, blend_mode='gaussian')
                crop_mattes = merge_image_overlap(matte_from_processed, positions, crop_size=args.nafn_patch_size,
                                                  resolution=(B, 1, crop.shape[2], crop.shape[3]), overlap_size=args.overlap_size, blend_mode='gaussian')
                processed_crops.append(crop_processed)
                processed_mattes.append(crop_mattes)
            
            outputs = merge_sliding_crops(processed_crops, crop_positions, original_width=W, overlap_size=args.overlap_size)
            mattes = merge_sliding_crops(processed_mattes, crop_positions, original_width=W, overlap_size=args.overlap_size)
            
            nafnet_output = net_1(outputs)
            nafnet_output = torch.clamp(nafnet_output, 0, 1)
            
            for i in range(nafnet_output.size(0)):
                out_path = os.path.join(args.output_dir, fname[i])
                torchvision.utils.save_image(nafnet_output[i].cpu(), out_path, normalize=True)
                print(f"Saved output image: {out_path}")
                
                if gts is not None:
                    psnr_val = compute_psnr(nafnet_output[i:i+1], gts[i:i+1])
                    ssim_val = compute_ssim(nafnet_output[i:i+1], gts[i:i+1])
                    lpips_val = lpips_fn(nafnet_output[i:i+1], gts[i:i+1]).mean()
                    
                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    total_lpips += lpips_val
                    count += 1
                    
                    print(f"  PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")
    
    e_time = time.time()
    print(f"\nElapsed time: {e_time - s_time} seconds")
    
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count
        print(f"\n=== Average Metrics ===")
        print(f"PSNR: {avg_psnr:.2f}")
        print(f"SSIM: {avg_ssim:.4f}")
        print(f"LPIPS: {avg_lpips:.4f}")

if __name__ == "__main__":
    args = parse_args()
    matte_predictor = ShadowMattePredictor(
        model_path=args.matte_generator_checkpoint,
        img_size=256,
        device=device
    )
    main()