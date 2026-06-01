from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def find_block_with_scale(prev_gray, next_gray, x1, y1, w, h, scale):
    """
    带缩放的块匹配：自动尝试稍微变大/变小
    返回 最佳位置 + 最小误差
    """
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 防止宽高变成 0 或 负数
    new_w = max(8, new_w)
    new_h = max(8, new_h)

    # 确保不越界
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x1 + new_w, prev_gray.shape[1])
    y2 = min(y1 + new_h, prev_gray.shape[0])

    template = prev_gray[y1:y2, x1:x2]
    result = cv2.matchTemplate(next_gray, template, cv2.TM_SQDIFF)
    min_val, _, top_left, _ = cv2.minMaxLoc(result)
    area = new_w * new_h
    avg_error = min_val / area
    return top_left, avg_error, new_w, new_h


def find_block_in_next_frame(prev_img_path, next_img_path, x1, y1, block_w, block_h, threshold=5000):
    # 1. 读图 + 灰度
    prev_img = cv2.imread(prev_img_path)
    next_img = cv2.imread(next_img_path)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    # 2. 尝试 3 种尺寸：原大小、稍微大、稍微小
    scales = [0.9, 1.0, 1.1]  # 缩小、正常、放大
    best_err = 99999999
    best_pos = (x1, y1)
    best_w = block_w
    best_h = block_h

    for s in scales:
        pos, avg_error, w, h = find_block_with_scale(prev_gray, next_gray, x1, y1, block_w, block_h, s)
        if avg_error < best_err:
            best_err = avg_error
            best_pos = pos
            best_w = w
            best_h = h

    # 3. 判断是否找到
    # found = best_err < threshold
    found=True
    tx, ty = best_pos

    return next_img, found, (tx, ty), best_err, best_w, best_h


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

def save_hot_pic_with_ellipse(frame_idx, score_array, ellipse_params, save_dir="heatmap_frames"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.imshow(score_array, cmap="gray", vmin=0, vmax=255)  # 改为灰度图
    save_path = os.path.join(save_dir, f"heatmap_{frame_idx:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

# ======================================================================================
# 🔥 1. 从记忆库生成【加权椭圆】，越新权重越大
# ======================================================================================
def ell_genorator_from_memory(ellipse_memory):
    if not ellipse_memory:
        return None

    # 总共只取最近 3 帧给 x/y 用
    memory = ellipse_memory  # 整体记忆还是保留10帧
    n = len(memory)

    # ====================
    # 🔥 w/h/angle：原来的全帧加权
    # ====================
    # weights_all = np.linspace(0.5, 1.0, n)
    # weights_all = weights_all / np.sum(weights_all)
    if n==0:
        weight=1
    else:
        weight=1/n

    w = h = angle = 0.0
    for i, (cxi, cyi, wi, hi, ai) in enumerate(memory):
        w += wi * weight
        h += hi * weight
        angle += ai * weight

    # ====================
    # 🔥 x/y：仅最近 3 帧，固定权重 0.5 / 0.3 / 0.2
    # ====================
    recent_3 = ellipse_memory[-3:]  # 只取最新3帧
    cx = cy = 0.0
    weight_x = [0.05, 0.15, 0.8][-len(recent_3):]  # 自动适配不足3帧

    for i, (cxi, cyi, _, _, _) in enumerate(recent_3):
        cx += cxi * weight_x[i]
        cy += cyi * weight_x[i]

    return (cx, cy, w, h, angle)

# ======================================================================================
# 🔥 2. 椭圆约束：内部100、中间*2、外部0
# ======================================================================================
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

    # 内外椭圆掩码
    mask_inner = gen_mask(cx, cy, w, h, angle, 0.7)   # 内椭圆
    mask_outer = gen_mask(cx, cy, w, h, angle, 1.3)   # 外椭圆
    mask_band = mask_outer & ~mask_inner  # 中间环带

    out = clipped_arr.copy()
    out[mask_inner] = 100                    # 内椭圆：最热
    out[~mask_outer] = 0                    # 外椭圆外：清零
    out[mask_band] = np.clip(out[mask_band] * 2, 0, 100)  # 中间乘2
    return out

# ======================================================================================
# 🔥 3. 判断相似 + 镜头切变
# ======================================================================================
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

def save_track_txt(track_results, save_path):
    sorted_frames = sorted(track_results.keys())
    with open(save_path, 'w') as f:
        for fid in sorted_frames:
            x, y, w, h = track_results[fid]
            f.write(f"{x},{y},{w},{h}\n")


# ======================================================================================
# 主程序
# ======================================================================================
if __name__ == "__main__":
    txt_save_path=r"C:\Users\21108\Desktop\MaskRefineSam\MaskRefineoutput\UWCOT220_000012.txt"
    mask_list = np.load(r"C:\Users\21108\Desktop\MaskRefineSam\logits_list\UWCOT220_000012.npy", allow_pickle=True)
    H, W = mask_list[0].shape
    ellipse_memory = []
    track_results={}

    gate_start = 10.0
    gate_end   = -100.0
    step       = -0.3

    weight_start = 1.0
    weight_end   = 0.0

    enhance_gate_min = -11.0
    enhance_gate_max = -5.0
    enhance_bias     = 0.6

    idx = -1
    for ori_mask in tqdm(mask_list, desc="Mask Refining:",total=len(mask_list)):
        idx += 1

        gate = gate_start
        second_floor = False
        third_floor = False
        last_area = 0
        score_array = np.zeros((H, W), dtype=np.float32)

        try_bool = ori_mask > 1.5
        non_zero_indices = np.argwhere(try_bool)
        if len(non_zero_indices) == 0:
            ellipse_memory = []
            # save_hot_pic_with_ellipse(idx, score_array, None)
            track_results[idx] = [0,0,0,0]
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

        # 生成记忆椭圆
        memory_ellipse_params = ell_genorator_from_memory(ellipse_memory)

        # 计算当前热力与椭圆
        clipped_arr, ellipse_params, cx, cy, used_thresh = get_robust_heat_center_and_ellipse(
            score_array, init_thresh=70, min_thresh=10
        )

        # 空保护
        if ellipse_params is None:
            ellipse_memory = []
            track_results[idx] = [0,0,0,0]
            # save_hot_pic_with_ellipse(idx, clipped_arr, None)
            continue

        # 相似判断
        if memory_ellipse_params is not None:
            is_similar, sudden_camera = similar_to_memory(ellipse_params, memory_ellipse_params)
            if is_similar:
                ellipse_memory.append(ellipse_params)
                clipped_arr = reduce_by_ellipse(clipped_arr, memory_ellipse_params)
            else:
                if sudden_camera:
                    ellipse_memory = []
                    clipped_arr = reduce_by_ellipse(clipped_arr, ellipse_params)
                else:
                    clipped_arr = reduce_by_ellipse(clipped_arr, memory_ellipse_params)
        else:
            ellipse_memory.append(ellipse_params)
            clipped_arr = reduce_by_ellipse(clipped_arr, ellipse_params)

        # ====================== 🔥 最终过滤：>75变白，<75变黑 ======================
        binary_mask = np.where(clipped_arr > 75, 255, 0).astype(np.uint8)
        # binary_mask 就是你 >75 过滤后的黑白掩码 (255=白, 0=黑)
        mask = binary_mask > 0  # 转成布尔掩码

        non_zero_indices = np.argwhere(mask)
        if len(non_zero_indices) == 0:
            bbox = [0, 0, 0, 0]
        else:
            y_min, x_min = non_zero_indices.min(axis=0).tolist()
            y_max, x_max = non_zero_indices.max(axis=0).tolist()
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        # 最终 bbox 格式 = [x_min, y_min, width, height]
        track_results[idx] = bbox


        # 保存黑白掩码
        # save_hot_pic_with_ellipse(idx, binary_mask, ellipse_params)

    save_track_txt(track_results, txt_save_path)