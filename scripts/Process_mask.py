from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# -----------------------------------------------------------------------------
# 1. 块匹配相关函数
# -----------------------------------------------------------------------------
def find_block_with_scale(prev_gray, next_gray, x1, y1, w, h, scale):
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = max(8, new_w)
    new_h = max(8, new_h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x1 + new_w, prev_gray.shape[1])
    y2 = min(y1 + new_h, prev_gray.shape[0])

    template_gray = prev_gray[y1:y2, x1:x2]
    result = cv2.matchTemplate(next_gray, template_gray, cv2.TM_SQDIFF)
    min_val, _, top_left, _ = cv2.minMaxLoc(result)
    area = new_w * new_h
    avg_error = min_val / area

    return top_left, min_val, new_w, new_h

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

    return next_img, prev_img, found, (tx, ty), best_err, best_w, best_h

# -----------------------------------------------------------------------------
# 2. 椭圆拟合相关函数
# -----------------------------------------------------------------------------
def get_principal_angle_by_projection(pts, angle_step=1.0):
    best_angle = 0.0
    max_span = 0.0
    for deg in np.arange(0, 180, angle_step):
        rad = np.deg2rad(deg)
        dx = np.cos(rad)
        dy = np.sin(rad)
        proj = pts[:,0] * dx + pts[:,1] * dy
        span = np.max(proj) - np.min(proj)
        if span > max_span:
            max_span = span
            best_angle = deg
    return best_angle

def get_robust_heat_center_and_ellipse(score_array, vmin=0, vmax=100, init_thresh=70, min_thresh=10):
    clipped = np.clip(score_array, vmin, vmax)
    current_thresh = init_thresh

    while current_thresh >= min_thresh:
        y_idx, x_idx = np.where(clipped >= current_thresh)
        if len(x_idx) >= 1:
            break
        current_thresh -= 1

    y_idx, x_idx = np.where(clipped >= current_thresh)
    if len(x_idx) == 0:
        return clipped, None, -1.0, -1.0, current_thresh

    hot_pts = np.column_stack([x_idx, y_idx])
    cx = np.mean(x_idx)
    cy = np.mean(y_idx)

    if len(hot_pts) < 5:
        return clipped, None, cx, cy, current_thresh

    angle = get_principal_angle_by_projection(hot_pts)
    w = np.max(x_idx) - np.min(x_idx)
    h = np.max(y_idx) - np.min(y_idx)

    ellipse_params = (cx, cy, w, h, angle)
    return clipped, ellipse_params, cx, cy, current_thresh

# -----------------------------------------------------------------------------
# 3. 可视化与保存函数
# -----------------------------------------------------------------------------
def save_hot_pic_with_ellipse(frame_idx, score_array, ellipse_params, save_dir="heatmap_frames"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.imshow(score_array, cmap="gray", vmin=0, vmax=255)
    save_path = os.path.join(save_dir, f"heatmap_{frame_idx:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_track_txt(track_results, save_path):
    sorted_frames = sorted(track_results.keys())
    with open(save_path, 'w') as f:
        for fid in sorted_frames:
            x, y, w, h = track_results[fid]
            f.write(f"{x},{y},{w},{h}\n")

def save_ellipse_centers(centers, save_path):
    with open(save_path, 'w') as f:
        for (cx, cy) in centers:
            f.write(f"{cx:.2f},{cy:.2f}\n")

# -----------------------------------------------------------------------------
# 4. 椭圆记忆库函数
# -----------------------------------------------------------------------------
def ell_genorator_from_memory(ellipse_memory):
    if not ellipse_memory:
        return None

    memory = ellipse_memory
    n = len(memory)

    if n == 0:
        weight = 1
    else:
        weight = 1 / n

    w = h = angle = 0.0
    for i, (cxi, cyi, wi, hi, ai) in enumerate(memory):
        w += wi * weight
        h += hi * weight
        angle += ai * weight

    recent_3 = ellipse_memory[-3:]
    cx = cy = 0.0
    weight_x = [0.05, 0.15, 0.8][-len(recent_3):]

    for i, (cxi, cyi, _, _, _) in enumerate(recent_3):
        cx += cxi * weight_x[i]
        cy += cyi * weight_x[i]

    return (cx, cy, w, h, angle)

# -----------------------------------------------------------------------------
# 5. 椭圆约束与相似度判断
# -----------------------------------------------------------------------------
def reduce_by_ellipse(clipped_arr, look_ellipse_params):
    if look_ellipse_params is None:
        return clipped_arr

    H, W = clipped_arr.shape
    cx, cy, w, h, angle = look_ellipse_params

    def gen_mask(cx, cy, w, h, angle, scale):
        y, x = np.indices((H, W))
        dx = x - cx
        dy = y - cy
        rad = np.deg2rad(angle)
        c, s = np.cos(rad), np.sin(rad)
        a = (w * scale) / 2.0 + 1e-6
        b = (h * scale) / 2.0 + 1e-6
        term1 = ((dx * c + dy * s) / a) ** 2
        term2 = ((-dx * s + dy * c) / b) ** 2
        return (term1 + term2) <= 1.0

    mask_inner = gen_mask(cx, cy, w, h, angle, 0.7)
    mask_outer = gen_mask(cx, cy, w, h, angle, 1.3)
    mask_band = mask_outer & ~mask_inner

    out = clipped_arr.copy()
    out[mask_inner] = 100
    out[~mask_outer] = 0
    out[mask_band] = np.clip(out[mask_band] * 2, 0, 100)
    return out

def similar_to_memory(ellipse_params, memory_ellipse_params):
    if ellipse_params is None or memory_ellipse_params is None:
        return False, True

    cx_n, cy_n, w_n, h_n, a_n = ellipse_params
    cx_m, cy_m, w_m, h_m, a_m = memory_ellipse_params

    shift = np.hypot(cx_n - cx_m, cy_n - cy_m)
    scale_ok = (0.4 < w_n/w_m < 2.5) and (0.4 < h_n/h_m < 2.5)
    ang_diff = min(abs(a_n - a_m), 180 - abs(a_n - a_m))

    is_similar = True
    sudden_camera = False

    if shift > 100 or not scale_ok or ang_diff > 30:
        is_similar = False
        if shift > 100:
            sudden_camera = True

    return is_similar, sudden_camera

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # ====================== 块匹配追踪初始化 ======================
    mt_center_list = []
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    mt_frame_dir    = r"C:\Users\21108\Desktop\MaskRefineSam\data\UW-COT220\UWCOT220_000012\imgs"
    mt_save_dir     = r"C:\Users\21108\Desktop\MaskRefineSam\output_frames"
    mt_block_save_dir = r"C:\Users\21108\Desktop\MaskRefineSam\output_frames\template_blocks"

    os.makedirs(mt_save_dir, exist_ok=True)
    os.makedirs(mt_block_save_dir, exist_ok=True)

    mt_frames       = sorted([f for f in os.listdir(mt_frame_dir) if f.endswith(('png', 'jpg'))])
    mt_length       = len(mt_frames)

    mt_X1, mt_Y1    = 403, 66
    mt_BLOCK_W      = 84
    mt_BLOCK_H      = 32

    # ====================== 掩码椭圆拟合初始化 ======================
    txt_save_path   = r"C:\Users\21108\Desktop\MaskRefineSam\MaskRefineoutput\UWCOT220_000012.txt"
    center_save_path = r"C:\Users\21108\Desktop\MaskRefineSam\lib\center_txt\MaskRefine.txt"
    mask_list       = np.load(r"C:\Users\21108\Desktop\MaskRefineSam\logits_list\UWCOT220_000012.npy", allow_pickle=True)
    H, W            = mask_list[0].shape

    ellipse_memory  = []
    track_results   = {}
    ellipse_centers = []

    # 阈值参数
    gate_start      = 10.0
    gate_end        = -100.0
    step            = -0.3
    weight_start    = 1.0
    weight_end      = 0.0
    enhance_gate_min = -11.0
    enhance_gate_max = -5.0
    enhance_bias    = 0.6

    idx = -1
    for ori_mask in tqdm(mask_list, desc="Mask Refining:", total=len(mask_list)):
        idx += 1

        # ====================== 块匹配追踪 ======================
        mt_current_path = os.path.join(mt_frame_dir, mt_frames[idx])
        mt_current_img  = cv2.imread(mt_current_path)

        if idx != 0:
            mt_prev_path = os.path.join(mt_frame_dir, mt_frames[idx - 1])
            mt_next_path = mt_current_path

            mt_res_img, mt_prev_img, mt_found, mt_center, mt_score, mt_w, mt_h = find_block_in_next_frame(
                mt_prev_path, mt_next_path, mt_X1, mt_Y1, mt_BLOCK_W, mt_BLOCK_H, threshold=30000
            )

            if mt_found:
                mt_tx, mt_ty = mt_center
                mt_cx = mt_tx + mt_w // 2
                mt_cy = mt_ty + mt_h // 2

                cv2.rectangle(mt_current_img, (mt_tx, mt_ty), (mt_tx + mt_w, mt_ty + mt_h), (0, 0, 255), 2)
                cv2.circle(mt_current_img, (mt_cx, mt_cy), 5, (0, 255, 255), -1)

                mt_X1, mt_Y1     = mt_tx, mt_ty
                mt_BLOCK_W       = mt_w
                mt_BLOCK_H       = mt_h
                mt_center_list.append((mt_cx, mt_cy))
            else:
                mt_center_list.append((0, 0))

            mt_prev_block = mt_prev_img[mt_Y1: mt_Y1 + mt_BLOCK_H, mt_X1: mt_X1 + mt_BLOCK_W]
            mt_block_save_path = os.path.join(mt_block_save_dir, f"template_block_idx_{idx:03d}.jpg")

        else:
            mt_center_list.append((mt_X1 + mt_BLOCK_W // 2, mt_Y1 + mt_BLOCK_H // 2))

        mt_save_path = os.path.join(mt_save_dir, f"frame_{idx:03d}.jpg")

        # ====================== 掩码椭圆拟合 ======================
        gate = gate_start
        second_floor = False
        third_floor = False
        last_area = 0
        score_array = np.zeros((H, W), dtype=np.float32)

        try_bool = ori_mask > 1.5
        non_zero_indices = np.argwhere(try_bool)
        if len(non_zero_indices) == 0:
            ellipse_memory = []
            track_results[idx] = [0,0,0,0]
            ellipse_centers.append((-1, -1))
            continue

        while True:
            gate += step
            mask_bool = ori_mask > gate
            area = np.sum(mask_bool)

            if gate < gate_end:
                break
            if np.all(mask_bool):
                break

            if last_area != 0:
                area_rate = area / last_area
                if area_rate > 1.2 and not second_floor and not third_floor:
                    second_floor = True
                elif area_rate > 4.5 and second_floor and not third_floor:
                    third_floor = True

            if -10.0 < gate < -1.7:
                second_floor = True
            if gate < -30.0:
                third_floor = True

            norm = (gate - gate_start) / (gate_end - gate_start)
            weight = weight_start + norm * (weight_end - weight_start)

            if second_floor:
                if enhance_gate_min <= gate <= enhance_gate_max:
                    weight += enhance_bias

            score_array[mask_bool] += weight
            last_area = area

        memory_ellipse_params = ell_genorator_from_memory(ellipse_memory)

        clipped_arr, ellipse_params, cx, cy, used_thresh = get_robust_heat_center_and_ellipse(
            score_array, init_thresh=70, min_thresh=10
        )

# *********************************************************************************************
# *********************************************************************************************
# *********************************************************************************************
# *********************************************************************************************

        # *********************************************************************************************
        # *********************************************************************************************
        # 🔥 核心耦合区：块匹配 + 椭圆拟合 强强融合
        # *********************************************************************************************
        # 空保护
        if ellipse_params is None:
            ellipse_memory = []
            track_results[idx] = [0, 0, 0, 0]
            ellipse_centers.append((-1, -1))
            continue

        # 1. 获取当前帧 椭圆原始中心、块匹配中心
        curr_ellipse_cx, curr_ellipse_cy = cx, cy
        # 取当前帧块匹配跟踪中心
        curr_block_cx, curr_block_cy = mt_center_list[idx]

        # 2. 初始化第一帧基准（首帧对齐两个中心）
        if idx == 0:
            ref_ellipse_x, ref_ellipse_y = curr_ellipse_cx, curr_ellipse_cy
            ref_block_x, ref_block_y = curr_block_cx, curr_block_cy

        # 3. 计算椭圆相对首帧的偏移（目标真实形变位移）
        ellipse_dx = curr_ellipse_cx - ref_ellipse_x
        ellipse_dy = curr_ellipse_cy - ref_ellipse_y

        # 4. 耦合融合：块匹配做底盘 + 椭圆偏移做精修
        fuse_cx = curr_block_cx + ellipse_dx
        fuse_cy = curr_block_cy + ellipse_dy

        # 5. 原有相似度、记忆库逻辑完全保留，只把「存入中心」换成耦合后的点
        if memory_ellipse_params is not None:
            is_similar, sudden_camera = similar_to_memory(ellipse_params, memory_ellipse_params)
            if is_similar:
                ellipse_memory.append(ellipse_params)
                clipped_arr = reduce_by_ellipse(clipped_arr, memory_ellipse_params)
                # 存入耦合后的融合中心
                ellipse_centers.append((fuse_cx, fuse_cy))
            else:
                if sudden_camera:
                    ellipse_memory = []
                    clipped_arr = reduce_by_ellipse(clipped_arr, ellipse_params)
                    ellipse_centers.append((fuse_cx, fuse_cy))
                else:
                    clipped_arr = reduce_by_ellipse(clipped_arr, memory_ellipse_params)
                    ellipse_centers.append((fuse_cx, fuse_cy))
        else:
            ellipse_memory.append(ellipse_params)
            clipped_arr = reduce_by_ellipse(clipped_arr, ellipse_params)
            ellipse_centers.append((fuse_cx, fuse_cy))
        # *********************************************************************************************
        # *********************************************************************************************
        # *********************************************************************************************
        # *********************************************************************************************

# *********************************************************************************************
# *********************************************************************************************
# *********************************************************************************************
# *********************************************************************************************

        binary_mask = np.where(clipped_arr > 75, 255, 0).astype(np.uint8)
        mask = binary_mask > 0

        non_zero_indices = np.argwhere(mask)
        if len(non_zero_indices) == 0:
            bbox = [0, 0, 0, 0]
        else:
            y_min, x_min = non_zero_indices.min(axis=0).tolist()
            y_max, x_max = non_zero_indices.max(axis=0).tolist()
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        track_results[idx] = bbox

    # ====================== 结果保存 ======================
    save_track_txt(track_results, txt_save_path)
    save_ellipse_centers(ellipse_centers, center_save_path)

    with open(r"C:\Users\21108\Desktop\MaskRefineSam\lib\center_txt\mp4.txt", 'w') as f:
        for (mt_cx, mt_cy) in mt_center_list:
            f.write(f"{mt_cx},{mt_cy}\n")
