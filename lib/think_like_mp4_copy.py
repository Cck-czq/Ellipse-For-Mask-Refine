import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def find_block_with_scale(prev_gray, next_gray, x1, y1, w, h, scale):
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = max(8, new_w)
    new_h = max(8, new_h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x1 + new_w, prev_gray.shape[1])
    y2 = min(y1 + new_h, prev_gray.shape[0])

    # 🔥 保存：上一帧用来匹配的原始宏块（彩色）
    template_gray = prev_gray[y1:y2, x1:x2]
    result = cv2.matchTemplate(next_gray, template_gray, cv2.TM_SQDIFF)
    min_val, _, top_left, _ = cv2.minMaxLoc(result)
    area = new_w * new_h
    avg_error = min_val / area

    return top_left, min_val, new_w, new_h

# =============================================================================
# 🔥 我修改了这里：返回 上一帧原图 + 宏块区域
# =============================================================================
def find_block_in_next_frame(prev_img_path, next_img_path, x1, y1, block_w, block_h, threshold=5000):
    prev_img = cv2.imread(prev_img_path)
    next_img = cv2.imread(next_img_path)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    scales = [1.0]
    best_err = 99999999
    best_pos = (x1, y1)
    best_w = block_w
    best_h = block_h

    for s in scales:
        pos, min_val, w, h = find_block_with_scale(prev_gray, next_gray, x1, y1, block_w, block_h, s)
        if min_val < best_err:
            best_err = min_val
            best_pos = pos
            best_w = w
            best_h = h

    found = True
    tx, ty = best_pos

    # =========================================================================
    # 🔥 关键：返回上一帧图片，用于外部保存【匹配用的宏块】
    # =========================================================================
    return next_img, prev_img, found, (tx, ty), best_err, best_w, best_h


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    mt_frame_dir = r"C:\Users\21108\Desktop\MaskRefineSam\data\UW-COT220\UWCOT220_000012\imgs"
    mt_save_dir = r"C:\Users\21108\Desktop\MaskRefineSam\output_frames"
    mt_block_save_dir = r"C:\Users\21108\Desktop\MaskRefineSam\output_frames\template_blocks"  # 🔥 宏块保存目录
    os.makedirs(mt_save_dir, exist_ok=True)
    os.makedirs(mt_block_save_dir, exist_ok=True)  # 🔥 自动创建

    mt_frames = sorted([f for f in os.listdir(mt_frame_dir) if f.endswith(('png', 'jpg'))])
    mt_length = len(mt_frames)

    mt_X1, mt_Y1 = 403, 66
    mt_BLOCK_W = 84
    mt_BLOCK_H = 32

    for idx in range(mt_length):
        mt_current_path = os.path.join(mt_frame_dir, mt_frames[idx])
        mt_current_img = cv2.imread(mt_current_path)

        if idx != 0:
            mt_prev_path = os.path.join(mt_frame_dir, mt_frames[idx-1])
            mt_next_path = mt_current_path

            # =================================================================
            # 🔥 调用函数，拿到 prev_img（上一帧原图）
            # =================================================================
            mt_res_img, mt_prev_img, mt_found, mt_center, mt_score, mt_w, mt_h = find_block_in_next_frame(
                mt_prev_path, mt_next_path, mt_X1, mt_Y1, mt_BLOCK_W, mt_BLOCK_H, threshold=30000
            )

            print(f"idx:{idx}     found:{mt_found}     pos:{mt_center}    w:{mt_w}    h:{mt_h}     err:{mt_score:.0f}")

            if mt_found:
                mt_tx, mt_ty = mt_center
                mt_cx = mt_tx + mt_w//2
                mt_cy = mt_ty + mt_h//2

                cv2.rectangle(mt_current_img, (mt_tx, mt_ty), (mt_tx+mt_w, mt_ty+mt_h), (0,0,255), 2)
                cv2.circle(mt_current_img, (mt_cx, mt_cy), 5, (0,255,255), -1)

                mt_X1, mt_Y1 = mt_tx, mt_ty
                mt_BLOCK_W = mt_w
                mt_BLOCK_H = mt_h

            # =================================================================
            # 🔥 保存【上一帧用来匹配的宏块】，按 idx 编号
            # =================================================================
            prev_block = mt_prev_img[mt_Y1 : mt_Y1 + mt_BLOCK_H, mt_X1 : mt_X1 + mt_BLOCK_W]
            block_save_path = os.path.join(mt_block_save_dir, f"template_block_idx_{idx:03d}.jpg")
            # cv2.imwrite(block_save_path, prev_block)

        # 保存当前帧结果
        mt_save_path = os.path.join(mt_save_dir, f"frame_{idx:03d}.jpg")
        # cv2.imwrite(mt_save_path, mt_current_img)

    print("✅ 全部处理完成！")