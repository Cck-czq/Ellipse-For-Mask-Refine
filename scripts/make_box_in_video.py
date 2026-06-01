import cv2
import numpy as np


def draw_bbox_on_video(
        txt_path,  # 每一帧bbox的txt路径
        video_path,  # 输入视频路径
        output_path,  # 输出视频路径
        fps=30,  # 输出视频帧率
        thickness=2,  # 框线粗细
        color=(0, 255, 0)  # 框颜色：绿色
):
    # 1. 读取所有帧的bbox（一行对应一帧，支持逗号分隔）
    bbox_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                bbox_list.append([0, 0, 0, 0])
                continue

            # 🔥 修复：按逗号分割，适配你的 396,58,107,53 格式
            parts = line.replace(' ', '').split(',')
            xmin, ymin, w, h = map(float, parts)
            xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
            bbox_list.append([xmin, ymin, w, h])

    # 2. 打开视频
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. 定义输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 防止越界
        if frame_idx < len(bbox_list):
            xmin, ymin, w, h = bbox_list[frame_idx]
            # 只有bbox有效才画
            if w > 0 and h > 0:
                xmax = xmin + w
                ymax = ymin + h
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)

        out.write(frame)
        frame_idx += 1

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 画框完成！输出视频：{output_path}")


# ======================
# 在这里配置你的路径
# ======================
if __name__ == "__main__":
    TXT_PATH = r"C:\Users\21108\Desktop\MaskRefineSam\MaskRefineoutput\UWCOT220_000012.txt"
    VIDEO_INPUT_PATH = r"C:\Users\21108\Desktop\MaskRefineSam\data\UW-COT220\UWCOT220_000012\UWCOT220_000012.mp4"
    VIDEO_OUTPUT_PATH = r"C:\Users\21108\Desktop\MaskRefineSam\MaskRefineoutput\UWCOT220_000012.mp4"

    draw_bbox_on_video(
        txt_path=TXT_PATH,
        video_path=VIDEO_INPUT_PATH,
        output_path=VIDEO_OUTPUT_PATH,
        fps=30
    )