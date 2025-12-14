"""Build vocabulary from training data and save to disk."""

import argparse
import json
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from model.dataset import CXRDataset


def build_and_save_vocabulary(
    metadata_path: str,
    root_dir: str,
    output_path: str,
    vocab_size: int = 10000
):
    """Build vocabulary from all data splits and save to JSON."""
    # Load datasets
    train = CXRDataset(metadata_path, root_dir, split='train')
    val = CXRDataset(metadata_path, root_dir, split='validation')
    test = CXRDataset(metadata_path, root_dir, split='test')
    
    # Collect texts
    all_texts = []
    for dataset in [train, val, test]:
        for idx in tqdm(range(len(dataset)), leave=False):
            try:
                all_texts.append(dataset[idx]['report_text'])
            except:
                continue
    
    # Build vocabulary
    word_counts = Counter()
    for text in all_texts:
        word_counts.update(re.findall(r'\b\w+\b', text.lower()))
    
    most_common = word_counts.most_common(vocab_size - 2)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab['<UNK>'] = 0
    vocab['<PAD>'] = 1
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    vocab_data = {
        'vocab': vocab,
        'vocab_size': vocab_size,
        'num_words': len(vocab),
        'unk_token_id': 0,
        'pad_token_id': 1
    }
    
    with open(output_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    print(f"Vocabulary saved: {len(vocab)} words -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Build vocabulary')
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='vocab/vocabulary.json')
    parser.add_argument('--vocab_size', type=int, default=10000)
    
    args = parser.parse_args()
    build_and_save_vocabulary(args.metadata_path, args.root_dir, args.output_path, args.vocab_size)


if __name__ == '__main__':
    main()
