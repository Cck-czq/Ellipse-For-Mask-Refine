import os
import gc
import cv2
import sys
import clip
import torch
import numpy as np
import os.path as osp
from pathlib import Path
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

HYDRA_FULL_ERROR=1
color = [(255, 0, 0)]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def save_track_txt(track_results, save_path):
    sorted_frames = sorted(track_results.keys())
    with open(save_path, 'w') as f:
        for fid in sorted_frames:
            x, y, w, h = track_results[fid]
            f.write(f"{x},{y},{w},{h}\n")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def main():
    file_dir = r"../"
    for i in range(1,220+1,1):
        try:
            dataset_root = os.path.join(file_dir, "data", "UW-COT220")
            output_root = os.path.join(file_dir, "logits_list")


            str_temp = f"{i:06d}"
            video_path = os.path.join(dataset_root, f"UWCOT220_{str_temp}", f"UWCOT220_{str_temp}.mp4")
            gt_path = os.path.join(dataset_root, f"UWCOT220_{str_temp}", "groundtruth_rect.txt")
            lang_path = os.path.join(dataset_root, f"UWCOT220_{str_temp}", "language.txt")

            logits_list=[]
            logits_list_save_path=os.path.join(file_dir,"logits_list",f"UWCOT220_{str_temp}.npy")
            output_txt_path = os.path.join(output_root, f"UWCOT220_{str_temp}.txt")
            exp_name = f"UWCOT220_{str_temp}"

            file_path = Path(output_txt_path)
            if file_path.exists():
                print(f"{exp_name}.txt已存在，跳过")
                # continue
            else:
                print(f"{exp_name}开始处理")

            device = "cuda"
            model_path = os.path.join(file_dir, "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")
            model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml"
            predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda")
            frames_or_path = prepare_frames_or_path(video_path)
            prompts = load_txt(gt_path)
            track_results = {}
            frame_rate = 30

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
                bbox, track_label = prompts[0]

                ## load language
                language_file = lang_path
                with open(language_file, 'r') as file:
                    language = file.readline().replace(".", "").replace("\n", "").lower()

                # Load the model
                model, _ = clip.load("ViT-B/32", device=device, jit=False)
                texts = [language]
                input_ids = clip.tokenize(texts, truncate=True).to(device)

                with torch.no_grad():
                    lang_embedds = model.encode_text(input_ids)
                    lang_embedds /= lang_embedds.norm(dim=-1, keepdim=True)

                _,_, masks = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    box=bbox,
                    lang_embedds=lang_embedds,
                )

                for frame_idx, object_ids, masks in predictor.propagate_in_video(state,lang_embedds=lang_embedds):
                    mask_to_vis = {}
                    bbox_to_vis = {}

                    for obj_id, mask in zip(object_ids, masks):
                        mask = mask[0].cpu().numpy()
                        logits_list.append(mask)
                        mask = mask > 0.0
                        non_zero_indices = np.argwhere(mask)
                        if len(non_zero_indices) == 0:
                            bbox = [0, 0, 0, 0]
                        else:
                            y_min, x_min = non_zero_indices.min(axis=0).tolist()
                            y_max, x_max = non_zero_indices.max(axis=0).tolist()
                            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        bbox_to_vis[obj_id] = bbox
                        mask_to_vis[obj_id] = mask

                        track_results[frame_idx] = bbox

            txt_save_path = output_txt_path
            save_track_txt(track_results, txt_save_path)
            np.save(logits_list_save_path,logits_list)
            del predictor, state
            gc.collect()
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    main()