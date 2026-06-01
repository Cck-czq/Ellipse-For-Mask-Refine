import cv2
import numpy as np

def merge_two_video_side_by_side_mp4(
    video1_path,    # 左边视频
    video2_path,    # 右边视频
    score_path,     # 每一帧的分数文件路径
    out_path,       # 输出 MP4 路径
    fps=30,         # MP4 帧率
    max_frames=5000
):
    # 打开两个视频
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 读取分数
    with open(score_path, 'r', encoding='utf-8') as f:
        scores = [line.strip() for line in f.readlines() if line.strip()]

    frame_count = 0
    out = None  # 视频写入器

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # 获取尺寸
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        target_h = min(h1, h2)

        # 等比例缩放
        frame1 = cv2.resize(frame1, (int(w1 * target_h / h1), target_h))
        frame2 = cv2.resize(frame2, (int(w2 * target_h / h2), target_h))

        # 左右拼接
        merge_frame = np.hstack([frame1, frame2])

        # ===================== 叠加分数 =====================
        if frame_count < len(scores):
            score_text = f"Score: {scores[frame_count]}"
        else:
            score_text = "Score: N/A"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 3
        text_color = (0, 255, 255)
        bg_color = (0, 0, 0)

        (text_w, text_h), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
        img_h, img_w = merge_frame.shape[:2]
        x_pos = (img_w - text_w) // 2
        y_pos = text_h + 15

        # 背景条
        cv2.rectangle(merge_frame,
                      (x_pos - 10, y_pos - text_h - 10),
                      (x_pos + text_w + 10, y_pos + 10),
                      bg_color, -1)

        # 写字
        cv2.putText(merge_frame, score_text,
                    (x_pos, y_pos),
                    font, font_scale, text_color, font_thickness)
        # =====================================================

        # 第一次初始化视频写入器
        if out is None:
            h, w = merge_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # 写入帧
        out.write(merge_frame)

        frame_count += 1
        if frame_count >= max_frames:
            break

    # 释放
    if out is not None:
        out.release()
        print(f"✅ MP4 保存完成：{out_path}")

    cap1.release()
    cap2.release()

# ==================== 配置 ====================
if __name__ == "__main__":
    VID_LEFT  = r"C:\Users\21108\Desktop\MaskRefineSam\gtoutput\UWCOT220_000012.mp4"
    VID_RIGHT = r"C:\Users\21108\Desktop\MaskRefineSam\MaskRefineoutput\UWCOT220_000012.mp4"
    SCORE_TXT = r"C:\Users\21108\Desktop\keyan\sam\MMOT_Evaluation_Toolkit\save.txt"
    OUT_MP4   = r"C:\Users\21108\Desktop\MaskRefineSam\scripts\look_same_time\result.mp4"

    merge_two_video_side_by_side_mp4(
        video1_path=VID_LEFT,
        video2_path=VID_RIGHT,
        score_path=SCORE_TXT,
        out_path=OUT_MP4,
        fps=60
    )