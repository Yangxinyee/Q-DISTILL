"""Evaluation script for student Q-Former model."""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from model.student import create_student_model
from model.losses import StudentLoss
from model.dataset import CXRDatasetMedGemma, collate_fn_student_medgemma
from model.teacher import create_teacher_model, SimpleTokenizer


@torch.no_grad()
def evaluate_student(student_model, teacher_model, dataloader, criterion, device,
                    use_classification=False, mask_report=False, distill_target='bi'):
    """Evaluate student model on distillation and classification metrics."""
    student_model.eval()
    teacher_model.eval()

    total_loss, total_mse, total_cosine_sim, num_batches = 0.0, 0.0, 0.0, 0
    all_student_features, all_teacher_features = [], []
    all_cosine_sims, all_predictions, all_labels = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        teacher_ids = batch.get('teacher_input_ids', input_ids).to(device)
        teacher_mask = batch.get('teacher_attention_mask', attention_mask).to(device)

        if mask_report:
            input_ids = torch.zeros_like(input_ids)
            attention_mask = torch.zeros_like(attention_mask)

        # Teacher forward
        use_bi = (distill_target == 'bi')
        teacher_outputs = teacher_model(images, teacher_ids, teacher_mask, compute_itm=use_bi)
        teacher_queries = teacher_outputs[4] if use_bi else teacher_outputs[0]
        teacher_features = teacher_queries.mean(dim=1)

        # Student forward
        student_features = student_model(images, input_ids, attention_mask,
                                         mask_report=mask_report, return_logits=False)

        # Classification
        priority_labels = batch.get('priority_labels')
        if use_classification and priority_labels is not None:
            priority_labels = priority_labels.to(device)
            student_logits = student_model(images, input_ids, attention_mask,
                                          mask_report=mask_report, return_logits=True)
            preds = student_logits.argmax(dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(priority_labels.cpu().numpy())

        # Loss
        loss_inputs = {'student_features': student_features, 'teacher_features': teacher_features}
        if use_classification and priority_labels is not None:
            loss_inputs['student_logits'] = student_logits
            loss_inputs['labels'] = priority_labels
        loss_dict = criterion(**loss_inputs)

        total_loss += loss_dict['total_loss'].item()
        total_mse += loss_dict['mse_loss'].item()
        total_cosine_sim += loss_dict['cosine_similarity'].item()
        num_batches += 1

        cosine_sim = (F.normalize(student_features, dim=1) * F.normalize(teacher_features, dim=1)).sum(dim=1)
        all_cosine_sims.extend(cosine_sim.cpu().numpy())
        all_student_features.append(student_features.cpu())
        all_teacher_features.append(teacher_features.cpu())

    all_student_features = torch.cat(all_student_features, dim=0)
    all_teacher_features = torch.cat(all_teacher_features, dim=0)
    student_norms = torch.norm(all_student_features, p=2, dim=1)
    teacher_norms = torch.norm(all_teacher_features, p=2, dim=1)

    results = {
        'total_loss': total_loss / num_batches,
        'mse_loss': total_mse / num_batches,
        'cosine_similarity': total_cosine_sim / num_batches,
        'student_feature_norm': student_norms.mean().item(),
        'teacher_feature_norm': teacher_norms.mean().item(),
        'cosine_sims': np.array(all_cosine_sims)
    }

    if use_classification and len(all_predictions) > 0:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        results['accuracy'] = accuracy_score(all_labels, all_predictions)
        results['f1_macro'] = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        results['confusion_matrix'] = confusion_matrix(all_labels, all_predictions).tolist()

    return results


def create_visualizations(results, output_dir, split, use_classification=False):
    """Create evaluation visualizations."""
    fig, axes = plt.subplots(1, 3 if use_classification else 2, figsize=(15 if use_classification else 10, 4))

    # Cosine similarity distribution
    ax = axes[0]
    sns.histplot(results['cosine_sims'], bins=50, ax=ax, color='blue', alpha=0.7)
    ax.axvline(np.mean(results['cosine_sims']), color='red', linestyle='--', label=f"Mean: {np.mean(results['cosine_sims']):.3f}")
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Cosine Similarity Distribution')
    ax.legend()

    # Loss summary
    ax = axes[1]
    metrics = ['MSE Loss', 'Cosine Sim']
    values = [results['mse_loss'], results['cosine_similarity']]
    bars = ax.bar(metrics, values, color=['blue', 'green'])
    ax.set_ylabel('Value')
    ax.set_title('Loss Metrics')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}',
                ha='center', va='bottom', fontsize=10)

    # Confusion matrix
    if use_classification and 'confusion_matrix' in results:
        ax = axes[2]
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f"Acc: {results['accuracy']:.3f}, F1: {results['f1_macro']:.3f}")

    plt.tight_layout()
    plt.savefig(output_dir / f'evaluation_{split}.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Student Q-Former')
    
    # Data
    parser.add_argument('--csv_path', type=str, default='data/metadata.csv')
    parser.add_argument('--medgemma_metadata_path', type=str, default='data/medgemma_reports/metadata.csv')
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--medgemma_report_dir', type=str, default='data/medgemma_reports')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test'])
    
    # Model
    parser.add_argument('--student_checkpoint', type=str, required=True)
    parser.add_argument('--teacher_checkpoint', type=str, default='outputs/teacher_focal/checkpoint_best.pt')
    parser.add_argument('--chexzero_path', type=str, default='checkpt/chexzero_best.pt')
    parser.add_argument('--vocab_path', type=str, default='vocab/vocabulary.json')
    
    # Evaluation
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation_student')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--mask_report', action='store_true')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load student checkpoint and config
    checkpoint = torch.load(args.student_checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    # Load external config if exists
    config_path = Path(args.student_checkpoint).parent / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config.update(json.load(f))

    student_state = checkpoint.get('model_state_dict', checkpoint)
    lambda_classification = config.get('lambda_classification', 0.0)
    use_classification = lambda_classification > 0
    mask_report = args.mask_report or config.get('mask_report', False)
    distill_target = config.get('distill_target', 'bi')

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.get('vocab_size', 10000), max_length=512)
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    tokenizer.load_vocab(str(vocab_path))

    # Create dataset
    dataset = CXRDatasetMedGemma(args.csv_path, args.medgemma_metadata_path, args.root_dir,
                                  args.medgemma_report_dir, split=args.split, use_priority_labels=use_classification)
    collate = lambda b: collate_fn_student_medgemma(b, tokenizer, use_real_reports_for_teacher=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True, collate_fn=collate)

    # Infer student model config from checkpoint
    num_queries = config.get('student_num_queries') or (
        student_state['qformer.queries'].shape[1] if 'qformer.queries' in student_state else 8)
    num_layers = config.get('student_num_layers') or 4
    num_classes = config.get('num_classes', args.num_classes)
    
    if 'classification_head.3.weight' in student_state:
        num_classes = student_state['classification_head.3.weight'].shape[0]

    # Create student model
    student_model = create_student_model(
        chexzero_path=args.chexzero_path,
        teacher_checkpoint=args.teacher_checkpoint,
        num_classes=num_classes,
        device=device,
        num_queries=num_queries,
        num_layers=num_layers
    )
    student_model.load_state_dict(student_state, strict=False)
    student_model.eval()

    # Load teacher config
    teacher_dir = Path(args.teacher_checkpoint).parent
    teacher_config_path = teacher_dir / 'config.json'
    if teacher_config_path.exists():
        with open(teacher_config_path) as f:
            tc = json.load(f)
        vocab_size = tc.get('vocab_size', 10000)
        num_queries_t = tc.get('num_queries', 12)
        num_layers_t = tc.get('num_layers', 4)
        num_heads_t = tc.get('num_heads', 12)
        num_classes_t = tc.get('num_classes', 2)
    else:
        vocab_size, num_queries_t, num_layers_t, num_heads_t, num_classes_t = 10000, 12, 4, 12, 2

    # Create teacher model
    teacher_model = create_teacher_model(
        chexzero_path=args.chexzero_path,
        vocab_size=vocab_size,
        num_queries=num_queries_t,
        device=device,
        num_layers=num_layers_t,
        num_heads=num_heads_t,
        num_classes=num_classes_t
    )

    # Load teacher weights
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')
    teacher_state = teacher_checkpoint.get('model_state_dict', teacher_checkpoint)
    teacher_model.load_state_dict(teacher_state, strict=False)
    
    # Load vocabulary for teacher
    teacher_model.text_tokenizer.load_vocab(str(vocab_path))
    teacher_model.eval()

    # Create loss
    lambda_cosine = config.get('lambda_cosine', 1.0)
    lambda_distill = config.get('lambda_distill', 1.0)
    criterion = StudentLoss(
        lambda_mse=lambda_distill,
        lambda_cosine=lambda_cosine,
        lambda_classification=lambda_classification,
        num_classes=num_classes
    )

    print(f"Evaluating {args.split} split ({len(dataset)} samples)")
    print(f"Distill target: {distill_target}, Classification: {use_classification}, Mask report: {mask_report}")

    results = evaluate_student(
        student_model, teacher_model, dataloader, criterion, device,
        use_classification=use_classification, mask_report=mask_report, distill_target=distill_target
    )

    # Print results
    print(f"\n{'='*50}")
    print(f"Results ({args.split}):")
    print(f"  MSE Loss: {results['mse_loss']:.6f}")
    print(f"  Cosine Similarity: {results['cosine_similarity']:.4f}")
    if use_classification and 'accuracy' in results:
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 Macro: {results['f1_macro']:.4f}")
    print(f"{'='*50}")

    # Visualizations
    create_visualizations(results, output_dir, args.split, use_classification)
    print(f"Plots saved to {output_dir}/evaluation_{args.split}.png")

    # Save results
    results_save = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in results.items() if k != 'cosine_sims'}
    results_save['num_samples'] = len(results['cosine_sims'])
    results_save['cosine_sim_mean'] = float(np.mean(results['cosine_sims']))
    
    with open(output_dir / f'results_{args.split}.json', 'w') as f:
        json.dump(results_save, f, indent=2)
    print(f"Results saved to {output_dir}/results_{args.split}.json")


if __name__ == '__main__':
    main()
