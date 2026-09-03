import os
import torch
import torch.multiprocessing as multiprocessing
from decord import VideoReader, cpu
import math
from transformers import CLIPFeatureExtractor, CLIPVisionModel
import numpy as np
from torch.nn import functional as F
import json
import argparse
import cv2
from tqdm import tqdm

def get_frame_indices(total_frames, original_fps, target_fps, num_frm, multiple=1):
    sample_fps = max(1, round(original_fps / target_fps))
    frame_idx = [i for i in range(0, total_frames, sample_fps)]
    if len(frame_idx) < num_frm:
        while len(frame_idx) % multiple != 0:
            frame_idx.append(0)
        # If we have fewer frames than num_frm, just return all the frames
        return frame_idx 
    scale = 1.0 * len(frame_idx) / num_frm
    uniform_idx = [round((i + 1) * scale - 1) for i in range(num_frm)]
    frame_idx = [frame_idx[i] for i in uniform_idx]
    return frame_idx

def extract_clip_embeddings(frames, vision_tower, feature_extractor):
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    
    inputs = feature_extractor(images=frames_rgb, return_tensors="pt", padding=True)
    inputs = {k: v.to(vision_tower.device) for k, v in inputs.items()}
    
    with torch.cuda.amp.autocast(): 
        features = vision_tower(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]  # [batch_size, hidden_size]
    
    return features.half()

def process_video(video_path, gpu_id, vision_tower, feature_extractor, batch_size=256):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    original_fps = vr.get_avg_fps()
    target_fps = 1
    
    indices = get_frame_indices(total_frames, original_fps, target_fps, num_frm=total_frames)
    frame_features = []
    
    for start_idx in range(0, len(indices), batch_size):
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]
        batch_frames = vr.get_batch(batch_indices).asnumpy()
        features = extract_clip_embeddings(batch_frames, vision_tower, feature_extractor)
        frame_features.append(features.cpu())
    
    frame_features = torch.cat(frame_features, dim=0)
    return frame_features

def extract_feature(args, gpu_id, video_list):
    model_path = 'openai/clip-vit-large-patch14-336'
    feature_extractor = CLIPFeatureExtractor.from_pretrained(model_path)
    vision_tower = CLIPVisionModel.from_pretrained(model_path).cuda(gpu_id)
    vision_tower.requires_grad_(False)
    vision_tower.eval()

    for video in tqdm(video_list, desc=f'GPU {gpu_id}处理中'):
        try:
            video_path = os.path.join(args.video_dir, video)
            features = process_video(video_path, gpu_id, vision_tower, feature_extractor)
            
            save_path = os.path.join(args.feature_dir, f"{video.split('.')[0].replace('/', '_')}.pth")
            torch.save(features, save_path)
            
            print(f"Finished processing: {video}, feature shape: {features.shape}")
        except Exception as e:
            print(f"Error processing video {video}: {str(e)}")
            continue

def get_processed_videos(feature_dir):
    processed = set()
    if os.path.exists(feature_dir):
        for filename in os.listdir(feature_dir):
            if filename.endswith('.pth'):
                video_name = filename.replace('.pth', '') + '.mp4'
                processed.add(video_name)
    return processed

def main_extract(args):
    num_gpus = torch.cuda.device_count()
    
    with open(args.json_path, 'r') as f:
        all_data = json.load(f)
        
    processed_videos = get_processed_videos(args.feature_dir)
    remaining_videos = list(set([video for video in all_data if video.split('.')[0].replace('/', '_') + '.mp4' not in processed_videos]))
    
    print(f"Total videos: {len(all_data)}")
    print(f"Processed videos: {len(processed_videos)}")
    print(f"Remaining videos: {len(remaining_videos)}")
    
    if not remaining_videos:
        print("All videos have been processed")
        return
    
    os.makedirs(args.feature_dir, exist_ok=True)
    
    chunk_size = len(remaining_videos) // num_gpus
    data_chunks = [remaining_videos[i:i + chunk_size] for i in range(0, len(remaining_videos), chunk_size)]
    
    if len(data_chunks) > num_gpus:
        data_chunks[-2].extend(data_chunks[-1])
        data_chunks.pop()

    processes = []
    
    for i in range(num_gpus):
        p = multiprocessing.Process(target=extract_feature, args=(args, i, data_chunks[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"{num_gpus} GPUs have been processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default='./video_data/', help='video file directory')
    parser.add_argument('--json_path', default='./llava_video_mp4_files.json', help='JSON file path')
    parser.add_argument('--feature_dir', default='./features_clip', help='feature output directory')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
    
    args = parser.parse_args()
    
    torch.multiprocessing.set_start_method('spawn')
    main_extract(args)