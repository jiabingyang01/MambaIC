import os
from PIL import Image

root = "/home/zhaorun/zichen/yjb/projects/CV/MambaIC/dataset/wildfire/train"

bad_images = []

def is_image_file(filename):
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]
    return any(filename.lower().endswith(ext) for ext in extensions)

for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
        if not is_image_file(fname):
            continue
        
        fpath = os.path.join(dirpath, fname)
        try:
            img = Image.open(fpath)
            img.verify()       # 检查图像是否损坏（结构）
            img = Image.open(fpath).convert("RGB")  # 检查能否正常加载
        except Exception as e:
            print(f"[BAD] {fpath}  ({e})")
            bad_images.append(fpath)

print("\n=== 检查完成 ===")
print(f"坏图像数量: {len(bad_images)}")

# 写入文件并删除坏图像
if bad_images:
    with open("bad_images_deleted.txt", "w") as f:
        for x in bad_images:
            f.write(x + "\n")
            try:
                os.remove(x)
                print(f"[DELETED] {x}")
            except Exception as e:
                print(f"[ERROR DELETE] {x} ({e})")

    print("\n坏图像列表已写入 bad_images_deleted.txt")
    print("坏图像已全部删除")
else:
    print("未发现坏图像。")
