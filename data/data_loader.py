"""
Data loaders for Safety-WaRP-LLM

Supports:
1. Safety data: do-not-answer dataset (harmful prompts)
2. Utility data: GSM8K dataset (math problems)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


class SafetyDataset(Dataset):
    """
    Do-Not-Answer dataset for safety alignment
    Contains harmful prompts and safe responses
    """
    
    def __init__(self, dataset, max_samples=None):
        """
        Args:
            dataset: HuggingFace dataset object
            max_samples: Maximum number of samples to use (None = all)
        """
        if max_samples:
            self.data = dataset.select(range(min(max_samples, len(dataset))))
        else:
            self.data = dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # do-not-answer dataset uses 'question' field for harmful prompts
        # and has various model responses (we'll use GPT4 response as safe response)
        return {
            'harmful_prompt': item['question'],
            'safe_response': item.get('GPT4_response', ''),  # Safe refusal response
            'id': item.get('id', idx)
        }


class UtilityDataset(Dataset):
    """
    GSM8K dataset for utility (math) tasks
    Contains math problems and solutions
    """
    
    def __init__(self, dataset, max_samples=None):
        """
        Args:
            dataset: HuggingFace dataset object
            max_samples: Maximum number of samples to use (None = all)
        """
        if max_samples:
            self.data = dataset.select(range(min(max_samples, len(dataset))))
        else:
            self.data = dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'question': item['question'],
            'answer': item['answer']
        }


def create_safety_dataloader(
    batch_size=4,
    max_samples=None,
    tokenizer=None,
    num_workers=0,
    shuffle=False
):
    """
    Create DataLoader for safety data (do-not-answer)
    
    Args:
        batch_size: Batch size
        max_samples: Maximum samples to load (None = all)
        tokenizer: Tokenizer object (for reference, not used in collate)
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader: PyTorch DataLoader for safety data
    """
    try:
        logger.info("Loading do-not-answer dataset...")
        
        # Load do-not-answer dataset from LibrAI (trust_remote_code 제거)
        dataset = load_dataset('LibrAI/do-not-answer', split='train')
        logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Create SafetyDataset wrapper
        safety_dataset = SafetyDataset(dataset, max_samples=max_samples)
        logger.info(f"✓ SafetyDataset created: {len(safety_dataset)} samples")
        
        # Create DataLoader
        dataloader = DataLoader(
            safety_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=safety_collate_fn
        )
        
        logger.info(f"✓ DataLoader created: {len(dataloader)} batches")
        return dataloader
        
    except Exception as e:
        logger.error(f"Failed to create safety dataloader: {str(e)}", exc_info=True)
        raise


def create_utility_dataloader(
    batch_size=4,
    max_samples=None,
    tokenizer=None,
    num_workers=0,
    shuffle=True
):
    """
    Create DataLoader for utility data (GSM8K)
    
    Args:
        batch_size: Batch size
        max_samples: Maximum samples to load (None = all)
        tokenizer: Tokenizer object (for reference, not used in collate)
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader: PyTorch DataLoader for utility data
    """
    try:
        logger.info("Loading GSM8K dataset...")
        
        # Load GSM8K dataset
        dataset = load_dataset('openai/gsm8k', 'main', split='train')
        logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Create UtilityDataset wrapper
        utility_dataset = UtilityDataset(dataset, max_samples=max_samples)
        logger.info(f"✓ UtilityDataset created: {len(utility_dataset)} samples")
        
        # Create DataLoader
        dataloader = DataLoader(
            utility_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=utility_collate_fn
        )
        
        logger.info(f"✓ DataLoader created: {len(dataloader)} batches")
        return dataloader
        
    except Exception as e:
        logger.error(f"Failed to create utility dataloader: {str(e)}", exc_info=True)
        raise


def safety_collate_fn(batch):
    """
    Collate function for safety dataset
    Returns list of prompts and responses (not tokenized yet)
    """
    harmful_prompts = [item['harmful_prompt'] for item in batch]
    safe_responses = [item['safe_response'] for item in batch]
    ids = [item['id'] for item in batch]
    
    return {
        'harmful_prompt': harmful_prompts,
        'safe_response': safe_responses,
        'id': ids
    }


def utility_collate_fn(batch):
    """
    Collate function for utility dataset
    Returns list of questions and answers (not tokenized yet)
    """
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    
    return {
        'question': questions,
        'answer': answers
    }


if __name__ == '__main__':
    # Test data loaders
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test safety dataloader
    print("\n" + "="*70)
    print("Testing Safety DataLoader")
    print("="*70)
    safety_loader = create_safety_dataloader(batch_size=2, max_samples=10, shuffle=False)
    for batch_idx, batch in enumerate(safety_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Prompts: {len(batch['harmful_prompt'])} items")
        if batch['harmful_prompt']:
            print(f"    Example: {batch['harmful_prompt'][0][:80]}...")
        print(f"  Responses: {len(batch['safe_response'])} items")
        if batch_idx >= 2:
            break
    
    # Test utility dataloader
    print("\n" + "="*70)
    print("Testing Utility DataLoader (GSM8K)")
    print("="*70)
    utility_loader = create_utility_dataloader(batch_size=2, max_samples=10, shuffle=True)
    for batch_idx, batch in enumerate(utility_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Questions: {len(batch['question'])} items")
        if batch['question']:
            print(f"    Example: {batch['question'][0][:80]}...")
        print(f"  Answers: {len(batch['answer'])} items")
        if batch_idx >= 2:
            break
    
    print("\n✓ Data loaders working correctly!")
