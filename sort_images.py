import os
import shutil
from pathlib import Path

def sort_images_by_reference():
    # ================= 配置路径 =================
    # 1. 参考数据集的基础路径 (读取图片名的地方)
    dataset_base_dir = "/home/zhaorun/zichen/yjb/projects/CV/MambaIC/dataset/wildfire"
    
    # 2. 混杂在一起的源图片路径 (MambaIC生成的所有图)
    source_img_dir = "/home/zhaorun/zichen/yjb/projects/CV/MambaIC/output/MambaIC/0.008/all"
    
    # 3. 目标输出的基础路径 (分类后存放的地方)
    output_base_dir = "/home/zhaorun/zichen/yjb/projects/CV/MambaIC/output/MambaIC/0.008"

    # ================= 定义任务 =================
    # 格式: ("数据集子文件夹名", "目标子文件夹名")
    tasks = [
        ("smoke", "smoke"),       # 任务1: 处理 smoke
        ("wildfire", "wildfire"), # 任务2: 处理 wildfire
        ("train", "train")        # 任务3: 处理 train
    ]

    # ================= 开始处理 =================
    print(f"🚀 开始整理图片...")
    print(f"📂 源图片池: {source_img_dir}")

    total_copied = 0
    total_missing = 0

    for subfolder, target_name in tasks:
        # 构建完整路径
        ref_dir = os.path.join(dataset_base_dir, subfolder)
        dest_dir = os.path.join(output_base_dir, target_name)

        print(f"\n------------------------------------------------")
        print(f"正在处理: {subfolder} -> {target_name}")
        print(f"参考目录: {ref_dir}")
        print(f"目标目录: {dest_dir}")

        # 检查参考目录是否存在
        if not os.path.exists(ref_dir):
            print(f"❌ 错误: 参考目录不存在: {ref_dir}")
            continue

        # 如果目标目录不存在，则创建
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            print(f"✅ 已创建目标目录: {dest_dir}")

        # 获取参考目录下的所有文件名
        # 过滤掉隐藏文件，只保留常见的图片格式
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        file_list = [f for f in os.listdir(ref_dir) if f.lower().endswith(valid_extensions)]
        
        count = 0
        missing_count = 0

        for filename in file_list:
            # 源文件路径 (从 output/all 中找)
            src_file_path = os.path.join(source_img_dir, filename)
            # 目标文件路径
            dest_file_path = os.path.join(dest_dir, filename)

            if os.path.exists(src_file_path):
                # 复制文件 (copy2 保留文件元数据)
                shutil.copy2(src_file_path, dest_file_path)
                count += 1
                if count % 100 == 0:
                    print(f"   已复制 {count} 张图片...", end='\r')
            else:
                # 尝试解决扩展名可能变化的问题 (例如原图是jpg，生成图变成了png)
                # 如果你的文件名完全一致，可以忽略这段逻辑
                name_without_ext = os.path.splitext(filename)[0]
                potential_png = os.path.join(source_img_dir, name_without_ext + ".png")
                
                if os.path.exists(potential_png):
                    shutil.copy2(potential_png, os.path.join(dest_dir, name_without_ext + ".png"))
                    count += 1
                else:
                    # print(f"⚠️ 未在源目录找到: {filename}") # 如果缺失文件太多，可以注释掉这行避免刷屏
                    missing_count += 1

        print(f"✅ 完成 {subfolder}。 成功复制: {count} 张, 缺失: {missing_count} 张")
        total_copied += count
        total_missing += missing_count

    print(f"\n================================================")
    print(f"🎉 全部完成！")
    print(f"总计复制: {total_copied}")
    print(f"总计缺失: {total_missing}")

if __name__ == "__main__":
    sort_images_by_reference()