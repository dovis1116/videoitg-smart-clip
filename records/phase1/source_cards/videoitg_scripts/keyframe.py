import os
import torch
import torch.multiprocessing as multiprocessing
from decord import VideoReader, cpu
import math
from transformers import CLIPFeatureExtractor,CLIPVisionModel
import numpy as np
from torch.nn import functional as F
import json
import argparse
import cv2

def get_resized_wh(width, height, max_size):
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
    else:
        new_width = width
        new_height = height
    return new_width, new_height

def check_pure(mtx):
    unique_elements = np.unique(mtx)
    return len(unique_elements) == 1

def extract_second(image_filename):
    return image_filename.split('/')[-1].replace('.png', '').split('_')[-1]

def calculate_clip_feature_sim_2(image_1, image_2):
    similarity = F.cosine_similarity(image_1, image_2, dim=0)
    return similarity

def process_video(feature_path, clip_num, motion):
    feature = torch.load(feature_path)
    frame_list = []
    max_frame = feature.shape[0]
    if clip_num == "all":
        frame_list = list(range(0, max_frame))
    else:
        print(clip_num)
        for clip in clip_num:
            frame_range = list(range(clip*5, clip*5+5))
            frame_range = [f for f in frame_range if f < max_frame]
            frame_list.extend(frame_range)

    if motion:
        return frame_list

    feature_list = []
    for index in frame_list:
        feature_list.append((index, feature[index]))
    
    selected_frame_list = [feature_list[0][0]]  # 添加第0帧作为初始anchor
    curr_idx = 1
    
    while curr_idx < len(feature_list):
        curr_feature = feature_list[curr_idx][1]
        prev_feature = feature_list[curr_idx-1][1]
        sim_with_prev = calculate_clip_feature_sim_2(curr_feature, prev_feature)
        
        if sim_with_prev < 0.90:
            next_idx = curr_idx + 1
            while next_idx < len(feature_list):
                next_feature = feature_list[next_idx][1]
                sim_with_curr = calculate_clip_feature_sim_2(curr_feature, next_feature)
                
                if sim_with_curr < 0.97:
                    selected_frame_list.append(feature_list[curr_idx][0])
                    curr_idx = next_idx
                    break
                next_idx += 1
                
            if next_idx >= len(feature_list):
                selected_frame_list.append(feature_list[curr_idx][0])
                break
        curr_idx += 1
    
    if selected_frame_list[-1] != feature_list[-1][0]:
        last_anchor_feature = feature[selected_frame_list[-1]]
        last_frame_feature = feature_list[-1][1]
        sim_with_last = calculate_clip_feature_sim_2(last_frame_feature, last_anchor_feature)
        if sim_with_last < 0.90:
            selected_frame_list.append(feature_list[-1][0])
            
    return selected_frame_list

def extract_feature(args, video_list):
    for item in video_list:
        try:
            feature_path = os.path.join(args.feature_dir, item['video'].split('.')[0].replace('/', '_') + '.pth')
            motion = "yes" in item['motion'].lower()
            
            if not isinstance(item['clip_num'], list):
                if 'yes' in item['existence'].lower() and 'none' in item['clip_num'].lower():
                    item['clip_num'] = 'all'
                elif 'none' in item['clip_num'].lower():
                    continue
                
            frame_list = process_video(feature_path, clip_num=item['clip_num'], motion=motion)
            
            result_path = str(item['id']) + ".json"
            result_path = os.path.join(args.output_dir, result_path)
            
            result = {
                "id": item['id'],
                "frame_num": frame_list,
                "clip_num": item['clip_num'],
                "video": item['video'],
                "question": item['question'],
                "answer": item['answer'],
                "motion": item['motion'],
                "existence": item['existence']
            }
            
            with open(result_path, 'w') as f:
                json.dump(result, f)
                
            print(f"Finished processing: {item['video']}")
        except Exception as e:
            print(f"Error processing video {item['video']}: {str(e)}")
            continue

def get_processed_videos(output_dir):
    processed = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.json'):
                with open(os.path.join(output_dir, filename), 'r') as f:
                    processed.add(filename.split(".")[0])
    return processed

def main_extract(args):
    num_gpus = torch.cuda.device_count()
    
    with open(args.json_path, 'r') as f:
        all_data = json.load(f)
        
    processed_videos = get_processed_videos(args.output_dir)
    remaining_videos = list([item for item in all_data if item['id'] not in processed_videos])
    
    print(f"Total videos: {len(all_data)}")
    print(f"Processed videos: {len(processed_videos)}")
    print(f"Remaining videos: {len(remaining_videos)}")
    
    if not remaining_videos:
        print("All videos have been processed")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    chunk_size = len(remaining_videos) // num_gpus
    data_chunks = [remaining_videos[i:i + chunk_size] for i in range(0, len(remaining_videos), chunk_size)]
    
    if len(data_chunks) > num_gpus:
        data_chunks[-2].extend(data_chunks[-1])
        data_chunks.pop()

    processes = []
    
    for i in range(num_gpus):
        p = multiprocessing.Process(target=extract_feature, args=(args, data_chunks[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"{num_gpus} GPUs have been processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', default='features_clip/', help='feature output directory')
    parser.add_argument('--json_path', default='./filtered_sampled_data.json', help='JSON file path')
    parser.add_argument('--output_dir', default='./sampled_motion_data_90_97', help='feature output directory')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
    
    args = parser.parse_args()
    
    torch.multiprocessing.set_start_method('spawn')
    main_extract(args)