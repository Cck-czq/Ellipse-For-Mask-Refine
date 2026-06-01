import cv2
import os
from tqdm import tqdm


def video_to_frames(
        video_path,
        save_dir="video_frames",
        prefix="frame",
        digit=6  # 帧序号位数 000001
):
    """
    视频拆分成图片并保存
    :param video_path: 你的视频路径
    :param save_dir: 保存图片的文件夹
    :param prefix: 图片前缀
    :param digit: 序号位数
    """
    # 创建保存文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"视频总帧数：{total_frames}  |  FPS：{fps:.2f}")

    # 逐帧读取 + 保存
    frame_idx = 0
    for _ in tqdm(range(total_frames), desc="拆帧中"):
        ret, frame = cap.read()
        if not ret:
            break

        # 保存路径
        save_path = os.path.join(save_dir, f"{prefix}_{frame_idx:0{digit}d}.jpg")
        cv2.imwrite(save_path, frame)
        frame_idx += 1

    cap.release()
    print(f"\n✅ 拆帧完成！全部保存在：{save_dir}")


# ====================== 你只需要改这里 ======================
if __name__ == "__main__":
    # 1. 你的视频路径
    VIDEO_PATH = r"C:\Users\21108\Desktop\MaskRefineSam\data\UW-COT220\UWCOT220_000012\UWCOT220_000012.mp4"

    # 2. 你要保存到的文件夹
    SAVE_DIR = r"C:\Users\21108\Desktop\MaskRefineSam\data\UW-COT220\UWCOT220_000012\imgs"
    # =========================================================

    video_to_frames(VIDEO_PATH, SAVE_DIR)