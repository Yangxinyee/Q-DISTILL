"""Score report priority using MedGemma with multi-GPU support.

Classifies chest X-ray images into 2 priority categories:
- 0: Non-Urgent (normal or moderate findings)
- 1: Urgent (severe findings requiring immediate attention)
"""

import os
import argparse
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
import subprocess
import sys
import re


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


def load_report(study_id: str, report_dir: Path) -> str:
    """Load radiology report."""
    for name in [f"{study_id}_report.txt", f"{study_id}.txt"]:
        path = report_dir / name
        if path.exists():
            return path.read_text(encoding='utf-8').strip()
    return "[Report not found]"


def score_priority(image_path: str, report_text: str, model, processor, max_tokens: int = 300) -> dict:
    """Classify image into priority category (0: Non-Urgent, 1: Urgent)."""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {'priority_label': -1, 'category': 'error', 'reasoning': str(e)}

    if report_text.startswith("["):
        return {'priority_label': -1, 'category': 'error', 'reasoning': report_text}

    prompt = f"""Analyze this chest X-ray image and classify it into one of 2 PRIORITY categories.

PRIMARY FOCUS: Carefully examine the IMAGE content (lung fields, heart size, pleural spaces, medical devices, bone structures).
SECONDARY REFERENCE: Use the radiology report below as supplementary information.

Radiology Report (for reference):
{report_text}

Instructions:
1. Analyze the chest X-ray image for:
   - Lung abnormalities (consolidation, opacities, effusions, pneumothorax)
   - Cardiac abnormalities (cardiomegaly, mediastinal widening)
   - Pleural abnormalities (effusion, thickening)
   - Medical device positioning (tubes, lines, pacemakers)
   - Bone abnormalities (fractures, lesions)

2. Classify based on clinical urgency:

Priority Categories (2 classes):
- 0 (NON-URGENT): Normal chest X-ray OR moderate findings that do not require immediate intervention.
  Examples: Stable cardiomegaly, mild pleural effusion, chronic changes, routine post-operative findings, no acute abnormality.

- 1 (URGENT): Severe or critical findings requiring immediate clinical attention.
  Examples: Tension pneumothorax, large pneumothorax, severe pulmonary edema, massive pleural effusion, 
  critical device malposition (misplaced endotracheal tube, central line in wrong position), 
  acute respiratory distress signs, widened mediastinum suggesting aortic pathology.

Provide your response in this EXACT format:
PRIORITY: [0 or 1]
CATEGORY: [non-urgent/urgent]
REASONING: [Brief explanation focusing on key imaging findings that support your classification]"""

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist performing urgent triage classification of chest X-ray images. Focus on identifying findings that require immediate clinical attention versus those that can be managed routinely."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        decoded = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    # Parse response
    label = -1
    category = 'unknown'
    reasoning = ''

    match = re.search(r'PRIORITY:\s*([01])', decoded, re.IGNORECASE)
    if match:
        label = int(match.group(1))

    match = re.search(r'CATEGORY:\s*(\S+)', decoded, re.IGNORECASE)
    if match:
        category = match.group(1).lower().replace('-', '_')

    match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', decoded, re.IGNORECASE | re.DOTALL)
    if match:
        reasoning = match.group(1).strip()

    # Fallback
    if label < 0:
        lower = decoded.lower()
        if any(x in lower for x in ['urgent', 'critical', 'severe', 'immediate', 'emergency']):
            label, category = 1, 'urgent'
        else:
            label, category = 0, 'non_urgent'

    return {'priority_label': max(0, min(1, label)), 'category': category, 'reasoning': reasoning, 'raw': decoded}


def process_shard(df, image_root, report_dir, output_dir, model, processor, max_tokens, gpu_id, resume=True):
    """Process dataset shard."""
    records = []
    existing = set()
    
    if resume and (output_dir / "priority_metadata.csv").exists():
        existing = set(pd.read_csv(output_dir / "priority_metadata.csv")['study_id'])
        print(f"[GPU {gpu_id}] Resuming: {len(existing)} existing")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"GPU {gpu_id}", position=gpu_id):
        study_id = row['study_id']
        if resume and study_id in existing:
            continue

        img_path = os.path.join(image_root, row['image_path'])
        if not os.path.exists(img_path):
            result = {'priority_label': -1, 'category': 'error', 'reasoning': 'Image not found'}
        else:
            result = score_priority(img_path, load_report(study_id, report_dir), model, processor, max_tokens)

        # Save JSON
        with open(output_dir / f"{study_id}_priority.json", 'w') as f:
            json.dump({'study_id': study_id, **result}, f, indent=2, ensure_ascii=False)

        records.append({
            'study_id': study_id,
            'priority_label': result['priority_label'],
            'category': result['category'],
            'split': row['split']
        })

    return records


def spawn_worker(gpu_id, num_gpus, args):
    """Spawn GPU worker process."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [sys.executable, __file__,
           '--csv_path', args.csv_path, '--image_root', args.image_root,
           '--report_dir', args.report_dir, '--output_dir', args.output_dir,
           '--max_tokens', str(args.max_tokens), '--gpu_id', str(gpu_id),
           '--num_gpus', str(num_gpus), '--device', 'cuda:0']
    if args.resume:
        cmd.append('--resume')
    
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description='Score report priority (2-class)')
    parser.add_argument('--csv_path', default='data/metadata.csv')
    parser.add_argument('--image_root', default='data')
    parser.add_argument('--report_dir', default='data/reports')
    parser.add_argument('--output_dir', default='data/priority_scores')
    parser.add_argument('--max_tokens', type=int, default=300)
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
                f = output_dir / f"priority_metadata_gpu{i}.csv"
                if f.exists():
                    all_df.append(pd.read_csv(f))
                    f.unlink()
            
            if all_df:
                merged = pd.concat(all_df, ignore_index=True)
                merged.to_csv(output_dir / "priority_metadata.csv", index=False)
                print(f"\nMerged {len(merged)} results")
                print(f"  Non-Urgent (0): {(merged['priority_label']==0).sum()}")
                print(f"  Urgent (1): {(merged['priority_label']==1).sum()}")
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
    records = process_shard(df, args.image_root, Path(args.report_dir), output_dir,
                           model, processor, args.max_tokens, args.gpu_id, args.resume)

    if records:
        results_df = pd.DataFrame(records)
        if args.num_gpus:
            results_df.to_csv(output_dir / f"priority_metadata_gpu{args.gpu_id}.csv", index=False)
        else:
            out_file = output_dir / "priority_metadata.csv"
            if args.resume and out_file.exists():
                results_df = pd.concat([pd.read_csv(out_file), results_df], ignore_index=True)
            results_df.to_csv(out_file, index=False)

    print(f"\nDone! {len(records)} classifications saved to {output_dir}")


if __name__ == "__main__":
    main()
