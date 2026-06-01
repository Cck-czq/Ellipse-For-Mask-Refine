import cv2
import numpy as np

def load_points(txt_path):
    """加载点坐标文件：每行 x,y"""
    pts = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = map(float, line.split(','))
            pts.append((int(x), int(y)))
    return pts

def draw_points_on_video(video_path, txt_a, txt_b, out_video_path, radius=6):
    # 加载两列点
    pts_a = load_points(txt_a)
    pts_b = load_points(txt_b)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化写视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    # 逐帧画点
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 画A点 红色
        if frame_idx < len(pts_a):
            x1, y1 = pts_a[frame_idx]
            cv2.circle(frame, (x1, y1), radius, (0, 0, 255), -1)

        # 画B点 蓝色
        if frame_idx < len(pts_b):
            x2, y2 = pts_b[frame_idx]
            cv2.circle(frame, (x2, y2), radius, (255, 0, 0), -1)

        out.write(frame)
        frame_idx += 1
        if frame_idx >= total_frames:
            break

    cap.release()
    out.release()
    print("✅ 生成完成：", out_video_path)

if __name__ == "__main__":
    # ========== 这里改成你自己路径 ==========
    video_path = r"C:\Users\21108\Desktop\MaskRefineSam\data\UW-COT220\UWCOT220_000012\UWCOT220_000012.mp4"
    txt_a_path = r"C:\Users\21108\Desktop\MaskRefineSam\lib\center_txt\MaskRefine.txt"
    txt_b_path = r"C:\Users\21108\Desktop\MaskRefineSam\lib\center_txt\mp4.txt"
    out_path = r"C:\Users\21108\Desktop\MaskRefineSam\lib\center_txt\draw_points_resultaaa.mp4"

    # 执行
    draw_points_on_video(video_path, txt_a_path, txt_b_path, out_path, radius=5)