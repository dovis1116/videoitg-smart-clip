#!/usr/bin/env python
"""Run the smallest end-to-end VideoITG frame-selection smoke test."""

import argparse
import json
import os
import time

import numpy as np


def frame_indices(total_frames, fps, target_fps, max_frames):
    stride = max(1, round(float(fps) / target_fps))
    candidates = list(range(0, total_frames, stride))
    if len(candidates) <= max_frames:
        return candidates
    scale = len(candidates) / max_frames
    return [candidates[round((i + 1) * scale - 1)] for i in range(max_frames)]


def pad_input(tokenizer, input_ids):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    return (
        torch.nn.utils.rnn.pad_sequence(
            [input_ids], batch_first=True, padding_value=pad_id
        ),
        pad_id,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--topk", type=int, default=32)
    args = parser.parse_args()

    import torch
    from decord import VideoReader, cpu
    from eagle.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from eagle.mm_utils import get_model_name_from_path, tokenizer_image_token
    from eagle.model.builder import load_pretrained_model

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    tokenizer, model, processor, _ = load_pretrained_model(
        args.model,
        None,
        get_model_name_from_path(args.model),
        device_map="cuda:0",
    )
    model.half().eval().to("cuda:0")
    model_loaded = time.perf_counter()

    reader = VideoReader(args.video, ctx=cpu(0), num_threads=4)
    indices = frame_indices(len(reader), reader.get_avg_fps(), 2, args.max_frames)
    frames = reader.get_batch(indices).asnumpy()
    video = processor.preprocess(frames, return_tensors="pt")["pixel_values"].half()
    video = video.to("cuda:0")
    prompt = (
        DEFAULT_IMAGE_TOKEN
        + "Which IMAX movie isn't in the video? A. The Hunger Games: B. Catching Fire; "
        + "C. The Dark Knight; D. Oppenheimer; E. Dune\n"
        + "Please respond with only the letter of the correct answer.\n"
    )
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    input_ids, pad_id = pad_input(tokenizer, input_ids)
    input_ids = input_ids.to("cuda:0")
    attention_mask = input_ids.ne(pad_id).to("cuda:0")
    preprocess_done = time.perf_counter()

    with torch.inference_mode():
        response = model(input_ids, attention_mask=attention_mask, images=[video])
        logits = response.logits[0].sigmoid().flatten()
        values, selected = torch.topk(logits, k=min(args.topk, logits.numel()))
    torch.cuda.synchronize()

    selected_positions = selected.detach().cpu().tolist()
    result = {
        "model": args.model,
        "video": args.video,
        "video_frames": len(reader),
        "video_fps": float(reader.get_avg_fps()),
        "sampled_frames": len(indices),
        "topk": args.topk,
        "selected_frame_indices": sorted(indices[i] for i in selected_positions),
        "topk_scores": [float(x) for x in values.detach().cpu().tolist()],
        "load_seconds": model_loaded - started,
        "preprocess_seconds": preprocess_done - model_loaded,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
