import cv2
import os
from pathlib import Path


def images_to_video(
        img_folder,  # 图片文件夹路径
        save_path,  # 视频保存路径（xxx.mp4）
        fps=12  # 视频帧率（默认30，改数字就行）
):
    # ------------------- 自动获取所有图片并排序 -------------------
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_paths.extend(list(Path(img_folder).glob(ext)))

    # 按文件名排序（你要的标签顺序）
    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        print("❌ 没找到任何图片！检查路径是否正确")
        return

    # ------------------- 获取第一张图片的尺寸 -------------------
    first_img = cv2.imread(str(img_paths[0]))
    height, width = first_img.shape[:2]

    # ------------------- 开始合成视频 -------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print(f"✅ 开始合成视频，共 {len(img_paths)} 张图片")
    for i, path in enumerate(img_paths):
        img = cv2.imread(str(path))
        if img is None:
            print(f"⚠️  跳过损坏图片：{path.name}")
            continue
        out.write(img)
        print(f"已处理：{i + 1}/{len(img_paths)}", end="\r")

    out.release()
    print(f"\n🎉 视频合成完成！保存到：{save_path}")


# ====================== 【你只需要改这里！】 ======================
if __name__ == "__main__":
    # 1. 你的图片文件夹（必须改成你自己的）
    IMG_FOLDER = r"C:\Users\21108\Desktop\MaskRefineSam\output_frames"

    # 2. 视频保存路径 + 名字（必须改成你自己的）
    SAVE_PATH = r"C:\Users\21108\Desktop\MaskRefineSam\scripts\videos\result.mp4"

    # 3. 帧率（默认30，不用动也行）
    FPS = 30

    # 开始合成
    images_to_video(IMG_FOLDER, SAVE_PATH, FPS)