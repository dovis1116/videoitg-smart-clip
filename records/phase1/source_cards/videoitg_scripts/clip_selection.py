import json
import os
import openai
import time
from openai import OpenAI
from multiprocessing import Pool
import re

client = OpenAI(base_url="", api_key="")


sys_prompt = """
### Task:
You are an expert in analyzing video clip descriptions. Your task is to select which clip or combination of clips is necessary to answer the given question, ensuring the selected clips effectively cover the content of both the question and the answer.

#### Guidelines:
- Carefully read the descriptions to determine which clip(s) provide relevant content for the question and the answer.
- Clip descriptions are in chronological order. Use clip number to locate clips based on time-related expressions (e.g., "at the beginning of the video" suggests a smaller clip number, while "at the end of the video" suggests a larger one).
- If the question asks about the existence of an object or event. The object/action may not exist, meaning you can't find the answer in the description, but the question might still provide some clues. You need to find the sentence closest to those clues. 
- If asked about the whole video description or overall atmosphere, you should return all clip numbers.
- If there are no clues in all descriptions and cannot answer the question, return "None.".
- If multiple clips provide similar descriptions of the content and any of them can be used to answer the question, only return the first two corresponding clips.
- First, determine if one clip can answer the question or if multiple clips are needed. Then, return a list containing the selected clip(s) and an explanation.
- **Important**: Avoid including unnecessary clips.

### Output Format:
1. Your output should be formed in a JSON file.
2. Only return the Python dictionary string.
For example:
{"explanation": "...", "clip_num": "One clip: [Clip-2]"}
{"explanation": "...", "clip_num": "Multiple clips: [Clip-1, Clip-7, Clip-8]"}
{"explanation": "...", "clip_num": "None."}
"""

user_prompt ='''
Based on the descriptions and the given question, please identify the necessary clip(s).
'''

def testOpenaiChatCompletions(query_system, query, version='gpt-4o-mini'):
    retries = 3
    for _ in range(retries):
        try:
            completion = client.chat.completions.create(
                model=version, 
                temperature=0.0,
                top_p=0.1,
                messages=[
                    {"role": "system", "content": query_system},
                    {"role": "user", "content": [                        
                        {"type": "text", "text": query},
                    ]}
                ]
            )
            match = re.search(r'\{.*?\}', completion.choices[0].message.content, re.DOTALL)
            if match:
                response_content = match.group(0)
            else:
                response_content = completion.choices[0].message.content
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f'错误: {e}')
                continue
            if 'clip_num' in result and 'explanation' in result:
                return result
            else:
                print("missing clip_number or explanation")
                continue
        except openai.RateLimitError as e:
            print(f'错误: {e}')
            time.sleep(30)
        except openai.OpenAIError as e:
            print(f'错误: {e}')
    return "Unsuccessful"

def format_clips_descriptions(data):
    """
    将json文件中的clips和descriptions格式化输出
    
    Args:
        json_path: json文件路径
    """
    clips = data['clip_description']
    
    output = "Given the clips and descriptions:\n"
    for i, description in enumerate(clips):
        output += f"  **Clip-{i}**: {description}\n\n"
        
    return output

def format_QA_pairs(qa):
    """
    将json文件中的QA pairs格式化输出
    
    Args:
        json_path: json文件路径
    """
    question = qa['question'].replace('<image>\n', '')
    output = f"Given the following question:\n"
    output += f"**Question**: {question}\n"
    # output += f"**Answer**: {qa['answer']}\n"
    return output

def process_video(qa, save_path):
    video = qa['video']
    result_path = str(qa['id'])

    question = format_QA_pairs(qa)
    caption = format_clips_descriptions(qa)
    input_prompt = user_prompt + caption + question
    result = testOpenaiChatCompletions(sys_prompt, input_prompt)
    if result != "Unsuccessful":
        with open(f"{save_path}/{result_path}.json", "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Processed video: {video}")

# Main script
if __name__ == '__main__':
    save_path = "./keyframe_gpt_mini"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_path = './processed.json'
    caption_path = 'all_clip_desc.json'

    with open(data_path, 'r', encoding='utf-8') as f:
        video_qa = json.load(f)

    with open(caption_path, 'r', encoding='utf-8') as f:
        all_caption = json.load(f)

    video_to_desc = {}
    for item in all_caption:
        video_to_desc[item['video']] = item['clip_description']

    new_qa_pairs = []

    for qa_pair in video_qa:
        video = qa_pair.get('video')
        
        if video in video_to_desc:
            qa_pair['clip_description'] = video_to_desc[video]
            new_qa_pairs.append(qa_pair)

    pool = Pool(processes=256) 
    print(f"Processing {len(new_qa_pairs)} videos...")
    for qa in new_qa_pairs:
        pool.apply_async(process_video, args=(qa, save_path))

    pool.close()
    pool.join()

