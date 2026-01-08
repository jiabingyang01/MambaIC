# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models import MambaIC
# import warnings
# import os
# import sys
# import math
# import argparse
# import time
# import gc
# from pytorch_msssim import ms_ssim
# from PIL import Image

# # 忽略警告
# warnings.filterwarnings("ignore")

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2).item()
#     if mse == 0:
#         return 100.0
#     return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     val = ms_ssim(a, b, data_range=1.).item()
#     if val >= 1.0:
#         return 100.0
#     return -10 * math.log10(1 - val)

# def compute_bpp(out_net):
#     size = out_net['x_hat'].size()
#     num_pixels = size[0] * size[2] * size[3]
#     return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
#               for likelihoods in out_net['likelihoods'].values()).item()

# def pad(x, p=512):
#     h, w = x.size(2), x.size(3)
#     # 强制对齐到 p (默认512) 的倍数，彻底解决 Mamba 窗口报错
#     new_h = (h + p - 1) // p * p
#     new_w = (w + p - 1) // p * p
#     padding_left = (new_w - w) // 2
#     padding_right = new_w - w - padding_left
#     padding_top = (new_h - h) // 2
#     padding_bottom = new_h - h - padding_top
    
#     x_padded = F.pad(
#         x,
#         (padding_left, padding_right, padding_top, padding_bottom),
#         mode="constant",
#         value=0,
#     )
#     return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

# def crop(x, padding):
#     return F.pad(
#         x,
#         (-padding[0], -padding[1], -padding[2], -padding[3]),
#     )

# def format_size(size_bytes):
#     if size_bytes >= 1024 * 1024:
#         return f"{size_bytes / (1024 * 1024):.2f} MB"
#     else:
#         return f"{size_bytes / 1024:.2f} KB"

# def tiled_inference(net, x, tile_size=1024, padding_align=512):
#     """
#     分块推理 (FP32 模式)
#     """
#     N, C, H, W = x.size()
#     x_hat_full = torch.zeros_like(x)
#     total_bits = 0
    
#     # 计算需要切多少块
#     h_steps = math.ceil(H / tile_size)
#     w_steps = math.ceil(W / tile_size)
    
#     for i in range(h_steps):
#         for j in range(w_steps):
#             # 1. 坐标计算
#             h_start = i * tile_size
#             w_start = j * tile_size
#             h_end = min(h_start + tile_size, H)
#             w_end = min(w_start + tile_size, W)
            
#             # 2. 取块
#             x_crop = x[:, :, h_start:h_end, w_start:w_end]
            
#             # 3. Padding (使用 512 对齐)
#             x_crop_padded, padding_info = pad(x_crop, padding_align)
            
#             # 4. 推理 (纯 FP32)
#             out_net = net.forward(x_crop_padded)
            
#             # 计算这个块产生的 bits
#             bpp_chunk = compute_bpp(out_net)
#             num_pixels_chunk_padded = x_crop_padded.size(2) * x_crop_padded.size(3)
#             total_bits += bpp_chunk * num_pixels_chunk_padded
            
#             # 5. Crop & 放入大图
#             x_recon = crop(out_net['x_hat'], padding_info)
#             x_recon.clamp_(0, 1) 
#             x_hat_full[:, :, h_start:h_end, w_start:w_end] = x_recon
            
#             # 6. 显存清理
#             del out_net, x_crop, x_crop_padded, x_recon
            
#     return x_hat_full, total_bits

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="MambaIC Final Eval.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
#     parser.add_argument("--data", type=str, required=True, help="Path to dataset")
#     parser.add_argument("--output_dir", type=str, help="Save reconstructed images path")
    
#     # === 新增 Lambda 参数用于记录 ===
#     parser.add_argument("--lambda", dest="lmbda", type=float, required=True, 
#                         help="Lambda value (e.g. 0.025) for logging and filename")
    
#     parser.add_argument("--N", type=int, default=128)
#     parser.add_argument("--M", type=int, default=320)
    
#     # === 策略参数 ===
#     parser.add_argument("--tiling_threshold", type=int, default=2000, 
#                         help="Images larger than this will use tiling.")
#     parser.add_argument("--tile_size", type=int, default=1024)
    
#     args = parser.parse_args(argv)
#     return args

# def main(argv):
#     args = parse_args(argv)
#     path = args.data
    
#     # 这里的 output_dir 是存图片的，不是存 log 的
#     if args.output_dir and not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)

#     img_list = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
#     if args.cuda:
#         device = 'cuda:0'
#         torch.backends.cudnn.benchmark = True
#     else:
#         device = 'cpu'
        
#     print(f"Loading Model (FP32 Mode)... N={args.N}, M={args.M}, Lambda={args.lmbda}")
#     net = MambaIC(depths=[2,2,9,2], N=args.N, M=args.M)
#     net = net.to(device)
#     net.eval()
    
#     if args.checkpoint:
#         print("Loading", args.checkpoint)
#         ckpt = torch.load(args.checkpoint, map_location=device)
#         state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
#         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#         net.load_state_dict(state_dict)

#     print("-" * 130)
#     print(f"{'Image Name':<25} | {'Orig Size':<10} -> {'Est Size':<10} | {'Ratio':<6} | {'BPP':<8} | {'PSNR':<8} | {'MS-SSIM':<8} | {'Mode':<6}")
#     print("-" * 130)

#     metrics = {'psnr': 0, 'msssim': 0, 'bpp': 0, 'time': 0, 'orig': 0, 'comp': 0}
#     count = 0

#     for img_name in img_list:
#         if args.cuda:
#             torch.cuda.empty_cache()
#             gc.collect()

#         img_path = os.path.join(path, img_name)
#         orig_file_size = os.path.getsize(img_path)
        
#         try:
#             img = Image.open(img_path).convert('RGB')
#         except:
#             continue

#         x = transforms.ToTensor()(img).unsqueeze(0).to(device)
#         H, W = x.size(2), x.size(3)
#         num_pixels = H * W
        
#         # 自动分块策略
#         use_tiling = (H > args.tiling_threshold) or (W > args.tiling_threshold)
#         mode_str = "Tiled" if use_tiling else "Single"

#         with torch.no_grad():
#             if args.cuda: torch.cuda.synchronize()
#             s = time.time()
            
#             if use_tiling:
#                 # 分块推理 (FP32 + padding 512)
#                 x_hat, total_bits = tiled_inference(net, x, tile_size=args.tile_size, padding_align=512)
#                 bpp = total_bits / num_pixels
#             else:
#                 # 直推 (FP32 + padding 512)
#                 x_padded, padding = pad(x, 512)
#                 out_net = net(x_padded)
#                 bpp = compute_bpp(out_net)
#                 x_hat = crop(out_net['x_hat'], padding)
#                 x_hat.clamp_(0, 1)

#             if args.cuda: torch.cuda.synchronize()
#             e = time.time()
            
#             psnr = compute_psnr(x, x_hat)
#             msssim = compute_msssim(x, x_hat)
#             est_size = (bpp * num_pixels) / 8.0
#             ratio = orig_file_size / est_size if est_size > 0 else 0

#             print(f'{img_name[:25]:<25} | '
#                   f'{format_size(orig_file_size):<10} -> {format_size(est_size):<10} | '
#                   f'{ratio:<6.2f} | '
#                   f'{bpp:<8.3f} | '
#                   f'{psnr:<8.2f} | '
#                   f'{msssim:<8.2f} | '
#                   f'{mode_str:<6}')
            
#             if args.output_dir:
#                 save_path = os.path.join(args.output_dir, img_name)
#                 transforms.ToPILImage()(x_hat.squeeze().cpu()).save(save_path)
            
#             metrics['psnr'] += psnr
#             metrics['msssim'] += msssim
#             metrics['bpp'] += bpp
#             metrics['time'] += (e - s)
#             metrics['orig'] += orig_file_size
#             metrics['comp'] += est_size
#             count += 1
            
#             del x_hat, x

#     if count > 0:
#         avg = {k: v / count for k, v in metrics.items()}
#         total_ratio = metrics['orig'] / metrics['comp'] if metrics['comp'] > 0 else 0
        
#         # 构造最终输出字符串
#         result_str = (
#             f"Average Results ({count} images):\n"
#             f"Lambda        : {args.lmbda}\n"
#             f"Avg Orig Size : {format_size(avg['orig'])}\n"
#             f"Avg Est Size  : {format_size(avg['comp'])}\n"
#             f"Avg Ratio     : {total_ratio:.2f}\n"
#             f"Avg PSNR      : {avg['psnr']:.2f} dB\n"
#             f"Avg MS-SSIM   : {avg['msssim']:.4f} dB\n"
#             f"Avg BPP       : {avg['bpp']:.3f} bpp\n"
#             f"Avg Time      : {avg['time']:.4f} s\n"
#         )
        
#         # 1. 打印到控制台
#         print("-" * 130)
#         print(result_str)
#         print("-" * 130)
        
#         # 2. 保存到文件 output/MambaIC/{lambda}.txt
#         log_dir = "output/MambaIC"
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
            
#         log_file = os.path.join(log_dir, f"{args.lmbda}.txt")
        
#         try:
#             with open(log_file, "w") as f:
#                 f.write(result_str)
#             print(f"Results saved to: {log_file}")
#         except Exception as e:
#             print(f"Error saving log file: {e}")

#     else:
#         print("No images found.")

# if __name__ == "__main__":
#     main(sys.argv[1:])


import torch
import torch.nn.functional as F
from torchvision import transforms
from models import MambaIC
import warnings
import os
import sys
import math
import argparse
import time
import gc
from pytorch_msssim import ms_ssim
from PIL import Image

# 忽略警告
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

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p=512):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0)
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))

def format_size(size_bytes):
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / 1024:.2f} KB"

def tiled_inference(net, x, tile_size=1024, padding_align=512):
    N, C, H, W = x.size()
    x_hat_full = torch.zeros_like(x)
    total_bits = 0
    h_steps = math.ceil(H / tile_size)
    w_steps = math.ceil(W / tile_size)
    
    for i in range(h_steps):
        for j in range(w_steps):
            h_start = i * tile_size
            w_start = j * tile_size
            h_end = min(h_start + tile_size, H)
            w_end = min(w_start + tile_size, W)
            x_crop = x[:, :, h_start:h_end, w_start:w_end]
            x_crop_padded, padding_info = pad(x_crop, padding_align)
            out_net = net.forward(x_crop_padded)
            bpp_chunk = compute_bpp(out_net)
            num_pixels_chunk_padded = x_crop_padded.size(2) * x_crop_padded.size(3)
            total_bits += bpp_chunk * num_pixels_chunk_padded
            x_recon = crop(out_net['x_hat'], padding_info)
            x_recon.clamp_(0, 1) 
            x_hat_full[:, :, h_start:h_end, w_start:w_end] = x_recon
            del out_net, x_crop, x_crop_padded, x_recon
            
    return x_hat_full, total_bits

def parse_args(argv):
    parser = argparse.ArgumentParser(description="MambaIC Resume Eval.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, help="Save reconstructed images path")
    parser.add_argument("--lambda", dest="lmbda", type=float, required=True, help="Lambda value")
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--tiling_threshold", type=int, default=2000)
    parser.add_argument("--tile_size", type=int, default=1024)
    args = parser.parse_args(argv)
    return args

def load_progress(csv_path):
    """
    读取已存在的 CSV 日志，恢复进度
    返回: 
    1. processed_files (set): 已处理的文件名集合
    2. restored_metrics (dict): 已处理数据的累加值
    3. count (int): 已处理数量
    """
    processed_files = set()
    metrics = {'psnr': 0, 'msssim': 0, 'bpp': 0, 'time': 0, 'orig': 0, 'comp': 0}
    count = 0
    
    if not os.path.exists(csv_path):
        return processed_files, metrics, count

    print(f"Resuming from log: {csv_path}")
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            # 跳过表头（如果第一行是 ImageName 开头）
            start_idx = 0
            if len(lines) > 0 and "ImageName" in lines[0]:
                start_idx = 1
            
            for line in lines[start_idx:]:
                parts = line.strip().split(',')
                if len(parts) < 7: continue
                
                # CSV 格式: Name, Orig, Est, BPP, PSNR, SSIM, Time
                img_name = parts[0]
                orig_size = float(parts[1])
                est_size = float(parts[2])
                bpp = float(parts[3])
                psnr = float(parts[4])
                msssim = float(parts[5])
                time_val = float(parts[6])
                
                processed_files.add(img_name)
                metrics['orig'] += orig_size
                metrics['comp'] += est_size
                metrics['bpp'] += bpp
                metrics['psnr'] += psnr
                metrics['msssim'] += msssim
                metrics['time'] += time_val
                count += 1
    except Exception as e:
        print(f"Error reading log file: {e}. Starting fresh might be safer.")
        return set(), {'psnr': 0, 'msssim': 0, 'bpp': 0, 'time': 0, 'orig': 0, 'comp': 0}, 0
        
    print(f"--> Already processed {count} images. Resuming...")
    return processed_files, metrics, count

def main(argv):
    args = parse_args(argv)
    path = args.data
    
    # 准备日志目录
    log_dir = "output/MambaIC"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 定义详细日志文件路径 (CSV)
    csv_path = os.path.join(log_dir, f"{args.lmbda}_details.csv")

    # === 断点续传核心逻辑 ===
    # 1. 加载之前的进度
    processed_files, metrics, count = load_progress(csv_path)

    # 2. 如果是新文件，写入表头
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("ImageName,OrigSize,EstSize,BPP,PSNR,MS-SSIM,Time\n")

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    img_list = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    # 加载模型
    if args.cuda:
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        
    print(f"Loading Model... Lambda={args.lmbda}")
    net = MambaIC(depths=[2,2,9,2], N=args.N, M=args.M)
    net = net.to(device)
    net.eval()
    
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)

    print("-" * 130)
    print(f"{'Image Name':<25} | {'Orig Size':<10} -> {'Est Size':<10} | {'Ratio':<6} | {'BPP':<8} | {'PSNR':<8} | {'MS-SSIM':<8} | {'Status':<6}")
    print("-" * 130)

    for img_name in img_list:
        # === 跳过已处理的图片 ===
        if img_name in processed_files:
            # 简单打印一下跳过信息，或者完全不打印
            # print(f"{img_name[:25]:<25} | ... Skipped (Already Done)")
            continue

        if args.cuda:
            torch.cuda.empty_cache()
            gc.collect()

        img_path = os.path.join(path, img_name)
        orig_file_size = os.path.getsize(img_path)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            continue

        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        H, W = x.size(2), x.size(3)
        num_pixels = H * W
        
        use_tiling = (H > args.tiling_threshold) or (W > args.tiling_threshold)

        with torch.no_grad():
            if args.cuda: torch.cuda.synchronize()
            s = time.time()
            
            if use_tiling:
                x_hat, total_bits = tiled_inference(net, x, tile_size=args.tile_size, padding_align=512)
                bpp = total_bits / num_pixels
            else:
                x_padded, padding = pad(x, 512)
                out_net = net(x_padded)
                bpp = compute_bpp(out_net)
                x_hat = crop(out_net['x_hat'], padding)
                x_hat.clamp_(0, 1)

            if args.cuda: torch.cuda.synchronize()
            e = time.time()
            time_cost = e - s
            
            psnr = compute_psnr(x, x_hat)
            msssim = compute_msssim(x, x_hat)
            est_size = (bpp * num_pixels) / 8.0
            ratio = orig_file_size / est_size if est_size > 0 else 0

            # 打印进度
            print(f'{img_name[:25]:<25} | '
                  f'{format_size(orig_file_size):<10} -> {format_size(est_size):<10} | '
                  f'{ratio:<6.2f} | '
                  f'{bpp:<8.3f} | '
                  f'{psnr:<8.2f} | '
                  f'{msssim:<8.2f} | '
                  f'{"Done":<6}')
            
            # === 立即写入 CSV (断点续传的关键) ===
            # 使用 'a' (append) 模式
            with open(csv_path, 'a') as f:
                f.write(f"{img_name},{orig_file_size},{est_size},{bpp},{psnr},{msssim},{time_cost}\n")
            
            if args.output_dir:
                save_path = os.path.join(args.output_dir, img_name)
                transforms.ToPILImage()(x_hat.squeeze().cpu()).save(save_path)
            
            # 累加内存中的统计数据 (为了最后打印平均值)
            metrics['psnr'] += psnr
            metrics['msssim'] += msssim
            metrics['bpp'] += bpp
            metrics['time'] += time_cost
            metrics['orig'] += orig_file_size
            metrics['comp'] += est_size
            count += 1
            
            del x_hat, x

    # 最终统计 (基于加载的旧数据 + 本次新跑的数据)
    if count > 0:
        avg = {k: v / count for k, v in metrics.items()}
        total_ratio = metrics['orig'] / metrics['comp'] if metrics['comp'] > 0 else 0
        
        result_str = (
            f"Average Results ({count} images):\n"
            f"Lambda        : {args.lmbda}\n"
            f"Avg Orig Size : {format_size(avg['orig'])}\n"
            f"Avg Est Size  : {format_size(avg['comp'])}\n"
            f"Avg Ratio     : {total_ratio:.2f}\n"
            f"Avg PSNR      : {avg['psnr']:.2f} dB\n"
            f"Avg MS-SSIM   : {avg['msssim']:.4f} dB\n"
            f"Avg BPP       : {avg['bpp']:.3f} bpp\n"
            f"Avg Time      : {avg['time']:.4f} s\n"
        )
        
        print("-" * 130)
        print(result_str)
        print("-" * 130)
        
        # 保存最终总结文件
        final_log_file = os.path.join(log_dir, f"{args.lmbda}.txt")
        with open(final_log_file, "w") as f:
            f.write(result_str)
        print(f"Final summary saved to: {final_log_file}")
        print(f"Detailed logs are in: {csv_path}")

    else:
        print("No images found.")

if __name__ == "__main__":
    main(sys.argv[1:])