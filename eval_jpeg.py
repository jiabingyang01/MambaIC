import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
import io

warnings.filterwarnings("ignore")

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    if mse == 0:
        return 100.0
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    val = ms_ssim(a, b, data_range=1.).item()
    if val >= 1.0:
        return 100.0
    return -10 * math.log10(1 - val)

def format_size(size_bytes):
    """将字节转换为易读的 KB 或 MB"""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / 1024:.2f} KB"

def parse_args(argv):
    parser = argparse.ArgumentParser(description="JPEG testing script with avg sizes.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--quality", type=int, default=75, help="JPEG Quality (1-95)")
    args = parser.parse_args(argv)
    return args

def jpeg_compress_decompress(x, quality, save_path=None):
    """
    模拟 JPEG 编解码
    返回: 
    1. 重建后的 Tensor
    2. 压缩后的字节数 (Bytes)
    """
    img_tensor = x.squeeze(0).cpu()
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_tensor)
    
    buffer = io.BytesIO()
    # 编码
    img_pil.save(buffer, format='JPEG', quality=quality)

    jpeg_bytes = buffer.getvalue()
    if save_path is not None:
        with open(save_path, "wb") as f:
            f.write(jpeg_bytes)

    # 获取压缩后的大小 (字节)
    size_bytes = len(jpeg_bytes)

    buffer.seek(0)
    # 解码
    img_rec = Image.open(buffer).convert('RGB')
    
    to_tensor = transforms.ToTensor()
    x_hat = to_tensor(img_rec).unsqueeze(0).to(x.device)
    
    return x_hat, size_bytes

def main(argv):
    args = parse_args(argv)
    path = args.data
    
    img_list = []
    if not os.path.exists(path):
        print(f"Error: Path {path} not found.")
        return

    output_root = "/home/zhaorun/zichen/yjb/projects/CV/MambaIC/output/JPEG"
    output_dir = os.path.join(output_root, str(args.quality))
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            img_list.append(file)
            
    if args.cuda and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
        
    print(f"Test Device: {device} | JPEG Quality: {args.quality}")
    print(f"{'Image Name':<25} | {'Orig Size':<10} -> {'Comp Size':<10} | {'Ratio':<6} | {'BPP':<8} | {'PSNR':<8} | {'MS-SSIM':<8}")
    print("-" * 110)

    count = 0
    PSNR_sum = 0
    MS_SSIM_sum = 0
    Bit_rate_sum = 0
    total_time = 0
    
    # 新增：用于统计平均大小
    orig_size_sum = 0
    comp_size_sum = 0

    for img_name in img_list:
        img_path = os.path.join(path, img_name)
        
        # 1. 获取硬盘上的原文件大小
        orig_file_size = os.path.getsize(img_path)
        
        # 读取图像
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Skipping {img_name}: {e}")
            continue

        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        num_pixels = x.size(2) * x.size(3)
        
        if args.cuda: torch.cuda.synchronize()
        s = time.time()
        
        # 2. 执行压缩，获取压缩后的大小 (comp_size_bytes)
        base_name = os.path.splitext(img_name)[0]
        out_path = os.path.join(output_dir, f"{base_name}.jpg")
        x_hat, comp_size_bytes = jpeg_compress_decompress(x, args.quality, save_path=out_path)
        
        if args.cuda: torch.cuda.synchronize()
        e = time.time()
        total_time += (e - s)

        x_hat.clamp_(0, 1)

        # 3. 计算指标
        cur_bpp = (comp_size_bytes * 8) / num_pixels
        cur_psnr = compute_psnr(x, x_hat)
        cur_msssim = compute_msssim(x, x_hat)
        
        # 计算单个文件的压缩比
        compression_ratio = orig_file_size / comp_size_bytes if comp_size_bytes > 0 else 0

        print(f'{img_name[:25]:<25} | '
              f'{format_size(orig_file_size):<10} -> {format_size(comp_size_bytes):<10} | '
              f'{compression_ratio:<6.2f} | '
              f'{cur_bpp:<8.3f} | '
              f'{cur_psnr:<8.2f} | '
              f'{cur_msssim:<8.2f}')

        # 4. 累加数据
        PSNR_sum += cur_psnr
        MS_SSIM_sum += cur_msssim
        Bit_rate_sum += cur_bpp
        orig_size_sum += orig_file_size
        comp_size_sum += comp_size_bytes
        count += 1

    if count > 0:
        # 计算各项平均值
        avg_psnr = PSNR_sum / count
        avg_msssim = MS_SSIM_sum / count
        avg_bit_rate = Bit_rate_sum / count
        avg_time = total_time / count
        
        avg_orig_size = orig_size_sum / count
        avg_comp_size = comp_size_sum / count
        
        # 整体平均压缩比 (总原大小 / 总压后大小)
        total_ratio = orig_size_sum / comp_size_sum if comp_size_sum > 0 else 0

        print("-" * 110)
        print(f'Average Results ({count} images):')
        print(f'Avg Orig Size : {format_size(avg_orig_size)}')
        print(f'Avg Comp Size : {format_size(avg_comp_size)}')
        print(f'Avg Ratio     : {total_ratio:.2f} (Total Orig / Total Comp)')
        print(f'Avg PSNR      : {avg_psnr:.2f} dB')
        print(f'Avg MS-SSIM   : {avg_msssim:.4f} dB')
        print(f'Avg Bit-rate  : {avg_bit_rate:.3f} bpp')
        print(f'Avg Time      : {avg_time:.4f} s')
    else:
        print("No images found.")

if __name__ == "__main__":
    main(sys.argv[1:])
