"""Generate medical reports using MedGemma with multi-GPU support."""

import os
import argparse
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import subprocess
import sys


def load_medgemma_model(device="cuda"):
    """Load MedGemma model."""
    model_id = "google/medgemma-27b-it"
    print(f"Loading MedGemma from {model_id}...")
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    print(f"Model loaded on {device}")
    return model, processor


def generate_report(image_path: str, model, processor, max_new_tokens: int = 512) -> str:
    """Generate medical report for a chest X-ray image."""
    try:
        with Image.open(image_path) as img:
            image = img.copy().convert('RGB')
    except Exception as e:
        return f"[ERROR: Failed to load image - {e}]"

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text", 
                    "text": "You are an expert Board-Certified Radiologist. Generate radiology reports directly without any conversational language, introductions, or explanatory text. Start immediately with the report content."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this Chest X-Ray image using a systematic 'Anatomy-Guided Chain-of-Thought' process.

First, perform an internal visual scan (do not output these steps, but use them to ensure accuracy):
1. Airway: Check trachea for deviation or narrowing.
2. Bones: Scan ribs, clavicles, and spine for fractures or lytic lesions.
3. Cardiac/Mediastinum: Assess heart size (CTR), aortic knob, and mediastinal contours.
4. Diaphragm: Check if costophrenic angles are sharp or blunted.
5. Lungs: Scan all zones for opacities, consolidation, nodules, masses, or pneumothorax.
6. Pleura: Look for thickening or effusion.
7. Soft Tissues/Lines: Check for subcutaneous emphysema and identify any medical devices.

Output the report directly. Do NOT include any introductory phrases like "Here's the report" or "Based on the image". Start immediately with "FINDINGS:" followed by the content. Output ONLY these two sections:

FINDINGS:
[Provide a detailed narrative description. Systematically describe the lungs, pleural spaces, cardiac silhouette, mediastinum, and osseous structures. Use precise medical terminology.]

IMPRESSION:
[Provide a concise, numbered summary of the diagnostic conclusions. If the exam is normal, state 'No acute cardiopulmonary process.']"""
                },
                {"type": "image", "image": image}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decoded = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    return decoded


def process_shard(df, image_root, output_dir, model, processor, max_new_tokens, gpu_id, resume=True):
    """Process dataset shard."""
    records = []

    if resume:
        existing = sum(1 for _, row in df.iterrows() 
                      if (output_dir / f"{row['study_id']}_medgemma_report.txt").exists())
        if existing > 0:
            print(f"[GPU {gpu_id}] Resuming: {existing} existing reports")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"GPU {gpu_id}", position=gpu_id):
        study_id = row['study_id']
        report_path = output_dir / f"{study_id}_medgemma_report.txt"

        if resume and report_path.exists():
            continue

        img_path = os.path.join(image_root, row['image_path'])
        if not os.path.exists(img_path):
            report_text = "[ERROR: Image not found]"
        else:
            report_text = generate_report(img_path, model, processor, max_new_tokens)

        report_path.write_text(report_text, encoding='utf-8')
        records.append({
            'study_id': study_id,
            'medgemma_report_path': f"{study_id}_medgemma_report.txt",
            'split': row['split']
        })

    return records


def spawn_worker(gpu_id, num_gpus, args):
    """Spawn GPU worker process."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [sys.executable, __file__,
           '--csv_path', args.csv_path, '--image_root', args.image_root,
           '--output_dir', args.output_dir, '--max_new_tokens', str(args.max_new_tokens),
           '--gpu_id', str(gpu_id), '--num_gpus', str(num_gpus), '--device', 'cuda:0']
    if args.resume:
        cmd.append('--resume')
    
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description='Generate reports using MedGemma')
    parser.add_argument('--csv_path', default='data/metadata.csv')
    parser.add_argument('--image_root', default='data')
    parser.add_argument('--output_dir', default='data/medgemma_reports')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} samples")

    # Multi-GPU mode
    if args.multi_gpu:
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            print("Only 1 GPU, using single GPU mode")
            args.multi_gpu = False
        else:
            print(f"Using {num_gpus} GPUs")
            processes = [spawn_worker(i, num_gpus, args) for i in range(num_gpus)]
            for p in processes:
                p.wait()

            # Merge results
            all_df = []
            for i in range(num_gpus):
                f = output_dir / f"metadata_gpu{i}.csv"
                if f.exists():
                    all_df.append(pd.read_csv(f))
                    f.unlink()
            
            if all_df:
                merged = pd.concat(all_df, ignore_index=True)
                merged.to_csv(output_dir / "metadata.csv", index=False)
                print(f"Merged {len(merged)} reports")
            return

    # Single GPU / Worker mode
    if args.gpu_id is not None and args.num_gpus is not None:
        shard_size = len(df) // args.num_gpus
        start = args.gpu_id * shard_size
        end = len(df) if args.gpu_id == args.num_gpus - 1 else start + shard_size
        df = df.iloc[start:end].reset_index(drop=True)
        print(f"GPU {args.gpu_id}: processing {start}-{end-1}")
    else:
        args.gpu_id = 0

    model, processor = load_medgemma_model(args.device)
    records = process_shard(df, args.image_root, output_dir, model, processor,
                           args.max_new_tokens, args.gpu_id, args.resume)

    if records:
        results_df = pd.DataFrame(records)
        if args.num_gpus:
            results_df.to_csv(output_dir / f"metadata_gpu{args.gpu_id}.csv", index=False)
        else:
            out_file = output_dir / "metadata.csv"
            if args.resume and out_file.exists():
                results_df = pd.concat([pd.read_csv(out_file), results_df], ignore_index=True)
                results_df = results_df.drop_duplicates(subset=['study_id'], keep='last')
            results_df.to_csv(out_file, index=False)

    print(f"\nDone! {len(records)} reports saved to {output_dir}")

    # Show sample
    if records:
        sample = output_dir / records[0]['medgemma_report_path']
        print(f"\nSample ({records[0]['study_id']}):\n{'='*60}")
        print(sample.read_text(encoding='utf-8')[:500])


if __name__ == "__main__":
    main()
