import json
import os
import time
import base64
import cv2
from multiprocessing import Pool

from openai import OpenAI

# Need to install decord in advance
from decord import VideoReader, cpu

# Initialize OpenAI client with empty credentials (to be filled by user)
client = OpenAI(base_url="", api_key="")

# Base path for videos (can be set if needed)
BASE_PATH = ""

# System prompt for GPT-4 Vision to analyze frame-QA relevance
sys_prompt = """
### Task:
You are an expert in analyzing the relationship between a video frame and a question-answer (QA) pair. Your task is to determine whether the content of a given video frame is relevant to the provided QA. If the frame content is relevant to the QA, output "Yes". If the frame content is not relevant, output "No".

#### Guidelines:
- Carefully read both the frame description and the QA (question and answer).
- "Relevant" means the frame provides information that helps answer the question or matches the answer.
- If the frame contains visual evidence or context supporting the QA, output "Yes".
- If the frame is unrelated to the QA or provides no useful information for answering, output "No".
- Only output "Yes" or "No". Do not provide explanations or extra text.

### Output Format:
Yes
or
No
"""

def load_video_base64(path, frame_indices):
    """
    Read specified frames from video and convert to base64 format
    
    Args:
        path (str): Path to the video file
        frame_indices (list): List of frame numbers to extract
        
    Returns:
        tuple: (video_id, base64Frames, real_indices)
            - video_id (str): Unique identifier for the video
            - base64Frames (list): List of base64 encoded frame images
            - real_indices (list): List of actual frame indices that were successfully loaded
    """
    # Initialize video reader with CPU context
    video = VideoReader(path, ctx=cpu(0), num_threads=1)
    
    # Generate video ID by removing base path and replacing slashes with underscores
    video_id = path.replace(BASE_PATH, "").split(".")[0].replace("/", "_")
    
    base64Frames = []
    real_indices = []
    MAX_SIZE = 4 * 1024 * 1024  # 4MB size limit for each frame
    
    for i in frame_indices:
        # Skip if frame index exceeds video length
        if i >= len(video):
            continue
            
        # Extract frame and convert from RGB to BGR (OpenCV format)
        frame = video[i]
        frame_bgr = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR)
        
        # Encode frame to PNG format and convert to base64
        _, buffer = cv2.imencode(".png", frame_bgr)
        buffer = base64.b64encode(buffer).decode("utf-8")
        
        # Resize frame if it exceeds size limit (4MB)
        while len(buffer.encode('utf-8')) > MAX_SIZE:
            width = int(frame_bgr.shape[1] * 0.9)
            height = int(frame_bgr.shape[0] * 0.9)
            frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode(".png", frame_bgr)
            buffer = base64.b64encode(buffer).decode("utf-8")
            
        base64Frames.append(buffer)
        real_indices.append(i)
        
    return video_id, base64Frames, real_indices

def testVisionOpenaiChatCompletions(base64Frames, query_system, query, version='gpt-4-vision-preview'):
    """
    Send vision request to OpenAI GPT-4 Vision API with retry mechanism
    
    Args:
        base64Frames (list): List of base64 encoded frame images
        query_system (str): System prompt for the API
        query (str): User query/question
        version (str): GPT model version to use
        
    Returns:
        OpenAI completion object or error string if failed
    """
    retries = 3
    for _ in range(retries):
        try:
            # Create completion request with vision capabilities
            completion = client.chat.completions.create(
                model=version,
                temperature=0,  # Deterministic output
                top_p=0.1,     # Low diversity for consistent results
                messages=[
                    {"role": "system", "content": query_system},
                    {"role": "user", "content": [
                        {"type": "text", "text": query},
                        *[
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{x}"}, "detail": "low"}
                            for x in base64Frames
                        ]
                    ]}
                ]
            )
            return completion
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(30)  # Wait 30 seconds before retry
            
    print(f"Failed after multiple retries.")
    return f"Unsuccessful: Failed after multiple retries."

def format_QA_pairs(qa):
    """
    Format question-answer pairs for GPT input
    
    Args:
        qa (dict): Dictionary containing question and answer
        
    Returns:
        str: Formatted string with question and answer
    """
    # Remove image placeholder from question
    question = qa['question'].replace('<image>\n', '')
    answer = qa.get('answer', '')
    
    output = f"Given the following question:\n"
    output += f"**Question**: {question}\n"
    output += f"**Answer**: {answer}\n"
    return output

def process_frame_qa(qa, save_path):
    """
    Process a single QA pair with video frames using GPT-4 Vision
    
    Args:
        qa (dict): Question-answer dictionary containing video path and frame numbers
        save_path (str): Directory path to save results
    """
    video_path = qa['video']
    frame_nums = qa.get('frame_num', [])
    result_path = str(qa['id'])
    
    # Load video frames and convert to base64
    try:
        video_id, base64Frames, real_indices = load_video_base64(video_path, frame_nums)
    except Exception as e:
        print(f"Failed to read video: {video_path}, error: {e}")
        return
        
    if not base64Frames:
        print(f"Empty video frames: {video_path}")
        return
        
    # Format QA pairs for GPT input
    input_prompt = format_QA_pairs(qa)
    
    # Send vision request to GPT-4
    completion = testVisionOpenaiChatCompletions(base64Frames, sys_prompt, input_prompt)
    
    # Prepare result dictionary
    result = {
        "id": qa['id'],
        "video": video_path,
        "question": qa['question'],
        "answer": qa.get('answer', ''),
        "frame_num": frame_nums,
        "frame_index": real_indices,
        "gpt_result": None,
        "gpt_raw": None
    }
    
    # Handle different completion result types
    if isinstance(completion, str) and completion.startswith("Unsuccessful"):
        result["gpt_result"] = "Unsuccessful"
        result["gpt_raw"] = completion
    else:
        try:
            # Extract content from successful completion
            gpt_content = completion.choices[0].message.content.strip()
            result["gpt_result"] = gpt_content
            result["gpt_raw"] = gpt_content
        except Exception as e:
            result["gpt_result"] = "Error"
            result["gpt_raw"] = str(e)
            
    # Save results to JSON file
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{result_path}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    print(f"Processed id: {qa['id']} video: {video_path}")

if __name__ == '__main__':
    # Main execution block
    # Load QA data from JSON file
    data_path = './videoitg_data/output.json'
    save_path = './frame_results'
    
    with open(data_path, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)
        
    print(f"Total {len(qa_list)} data entries, starting processing...")
    
    # Use multiprocessing pool for parallel processing
    # Adjust number of processes based on GPU memory and bandwidth
    pool = Pool(processes=64)
    
    # Submit all QA pairs for processing
    for qa in qa_list:
        pool.apply_async(process_frame_qa, args=(qa, save_path))
        
    # Wait for all processes to complete
    pool.close()
    pool.join()
