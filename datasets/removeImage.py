import shutil
import os

def copy_images_from_txt(root_path, txt_file, output_folder):
    if "VID" in root_path:
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        # 打开包含图片路径的txt文件
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            # 逐行复制图片
            for line in lines:
                image_path = line.strip()  # 去除每行末尾的换行符等空白字符
                # 将train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000 1 50 300处理成图片路径
                parts = image_path.split()
                dir = parts[0]
                fileName = parts[2].zfill(6) + '.JPEG'
                image_path = os.path.join(root_path, dir, fileName).replace("\\", "/")

                if os.path.exists(image_path):  # 检查路径是否存在
                    if not os.path.exists(output_folder + "/" + dir):
                        os.makedirs(output_folder + "/" + dir)
                    # 构建输出路径
                    output_path = os.path.join(output_folder, dir, fileName).replace("\\", "/")
                    # 复制图片文件
                    shutil.copy(image_path, output_path)
                    print(f"Copied {image_path} to {output_path}")
                else:
                    print(f"File not found: {image_path}")

    if "DET" in root_path:
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        # 打开包含图片路径的txt文件
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            # 逐行复制图片
            for line in lines:
                image_path = line.strip()  # 去除每行末尾的换行符等空白字符
                # 将train/ILSVRC2014_train_0000/ILSVRC2014_train_00000663 1处理成图片路径
                parts = image_path.split()[0].split("/")
                if len(parts) == 3:  # train/ILSVRC2014_train_0000/ILSVRC2014_train_00000663
                    dir = parts[0] + "/" + parts[1]
                    fileName = parts[2] + ".JPEG"
                else:  #
                    # 长度=4：train/ILSVRC2013_train/n01756291/n01756291_16993 1
                    dir = parts[0] + "/" + parts[1] + "/" + parts[2]
                    fileName = parts[3] + ".JPEG"

                image_path = os.path.join(root_path, dir, fileName)

                if os.path.exists(image_path):  # 检查路径是否存在
                    if not os.path.exists(output_folder + "/" + dir):
                        os.makedirs(output_folder + "/" + dir)
                    # 构建输出路径
                    output_path = os.path.join(output_folder, dir, fileName)
                    # 复制图片文件
                    shutil.copy(image_path, output_path)
                    print(f"Copied {image_path} to {output_path}")
                else:
                    print(f"File not found: {image_path}")

# 指定包含图片路径的txt文件和输出文件夹路径 (测试成功)
# root_path = r"F:\备份\data\ILSVRC\Data\VID"
# txt_path = r"D:\pro\DiffusionVID\datasets\ILSVRC2015\ImageSets\VID_train_15frames.txt"
# output_path = r"F:\备份\data\ILSVRC\Data\VID_new"

# VID_train_15frames (成功 2.7G)
# root_path = r"/root/autodl-tmp/data/ILSVRC/Data/VID"
# txt_path = r"/root/pro/DiffusionVID/datasets/ILSVRC2015/ImageSets/VID_train_15frames.txt"
# output_path = r"/root/autodl-tmp/data/ILSVRC/Data/VID_train_15frames"

# VID_train_every10frames.txt (成功 5.2G)
# root_path = r"/root/autodl-tmp/data/ILSVRC/Data/VID"
# txt_path = r"/root/pro/DiffusionVID/datasets/ILSVRC2015/ImageSets/VID_train_every10frames.txt"
# output_path = r"/root/autodl-tmp/data/ILSVRC/Data/VID_train_every10frames"

# val不用修改(8G)

# DET_train_30classes.txt (成功 6.2G)
root_path = r"/root/autodl-tmp/data/ILSVRC/Data/DET"
txt_path = r"/root/pro/DiffusionVID/datasets/ILSVRC2015/ImageSets/DET_train_30classes.txt"
output_path = r"/root/autodl-tmp/data/ILSVRC/Data/DET_train_30classes"

# 复制图片
copy_images_from_txt(root_path, txt_path, output_path)
