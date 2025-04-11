import argparse
import gc
import math
import json
import os
import time
import random
from contextlib import nullcontext
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cuda import flash_sdp_enabled
from torch.nn.attention import SDPBackend
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tiktoken
from torch.amp import autocast, GradScaler


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # Will be set based on tokenizer
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    use_rope: bool = True
    gradient_checkpointing: bool = True


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) implementation"""

    def __init__(self, dim, max_seq_len=None, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer('cos_cached', torch.empty(0), persistent=False)
        self.register_buffer('sin_cached', torch.empty(0), persistent=False)
        self._cache_initialized = False

    def _compute_cos_sin(self, device):
        if self._cache_initialized:
            return

        t = torch.arange(self.max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        cos = emb.cos()[None, None, :, :]  # Shape: (1, 1, seq_len, dim)
        sin = emb.sin()[None, None, :, :]  # Shape: (1, 1, seq_len, dim)
        self.cos_cached = cos
        self.sin_cached = sin
        self._cache_initialized = True

    def forward(self, x, seq_len=None):
        if not self._cache_initialized or self.cos_cached.device != x.device:
            self._compute_cos_sin(x.device)
        if seq_len is None:
            seq_len = x.shape[-2]  # shape is [batch, heads, seq_len, head_dim]
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}"
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head

        # Combined projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

        self.rotary_emb = RotaryEmbedding(self.head_size, max_seq_len=config.block_size) if config.use_rope else None
        self.use_flash_attention = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)

        # Reshape to (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Apply rotary embeddings if enabled
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(q, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.use_flash_attention:
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,  # let flash handle causal masking
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True,
                )
        else:
            # fallback to slower, default impl
            scale = 1.0 / math.sqrt(self.head_size)
            att = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = self.mask[:, :, :T, :T]
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = torch.matmul(att, v)
            del att

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side
        y = self.c_proj(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        # Make SwiGLU intermediate dimension multiple of 256 for efficiency, see Llama2 paper
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = ((n_hidden + 256 - 1) // 256) * 256

        self.w1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.w3 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.w2 = nn.Linear(n_hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.gradient_checkpointing = config.gradient_checkpointing

    def _attn_forward(self, x):
        return self.attn(self.ln_1(x))

    def _mlp_forward(self, x):
        return self.mlp(self.ln_2(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm structure (common in modern transformers like Llama)
        attn_output = self.attn(self.ln_1(x))
        x = x + attn_output

        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            token_embeddings=nn.Embedding(config.vocab_size, config.n_embd),
            # Positional embeddings (only used if not using RoPE)
            positional_embeddings=nn.Embedding(config.block_size, config.n_embd) if not config.use_rope else None,
            dropout=nn.Dropout(config.dropout),
            blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.token_embeddings.weight = self.lm_head.weight
        self.apply(self._init_weights)
        print(f"Model parameters: {self.get_num_params()/1e6:.2f}M")


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.use_rope is False and self.transformer.positional_embeddings is not None:
            n_params -= self.transformer.positional_embeddings.weight.numel()
        return n_params

    def _init_weights(self, module):
        # Initialization strategy inspired by Llama 2
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)) # Scale std dev
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # LayerNorm init is handled by PyTorch defaults (weight=1, bias=0) which is standard

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        B, T = tokens.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        # Get token embeddings
        x = self.transformer.token_embeddings(tokens) # (B, T, n_embd)

        # Add positional embeddings if not using RoPE
        if not self.config.use_rope and self.transformer.positional_embeddings is not None:
            positions = torch.arange(0, T, dtype=torch.long, device=tokens.device) # shape (T)
            pos_emb = self.transformer.positional_embeddings(positions) # shape (T, n_embd)
            x = x + pos_emb

        x = self.transformer.dropout(x)

        # Apply transformer blocks
        for block in self.transformer.blocks:
            x = block(x)

        # Apply final layer norm (Pre-LN structure applies LN before head)
        x = self.transformer.ln_f(x)

        # Get logits
        # If targets are provided, only compute logits for the positions where targets exist
        if targets is not None:
             # Helps save computation and memory, esp during training
            logits = self.lm_head(x) # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # Use ignore_index if needed for padding
        else:
            # During inference, only compute logit for the last token
            # Assuming tokens is (B, T), we take the embeddings for the last token
            logits = self.lm_head(x[:, [-1], :]) # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad() # Ensure no gradients are computed during generation
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text with controlled sampling"""
        self.eval() # Set model to evaluation mode
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass to get logits for the very last token
            logits, _ = self(idx_cond) # Will only compute for the last token due to optimization in forward
            logits = logits[:, -1, :] # Pluck the logits for the final token

            # Apply temperature scaling
            if temperature > 0:
                 logits = logits / temperature
            else: # Handle temperature=0 case (greedy decoding)
                 # Find the single max logit
                 _, idx_next = torch.topk(logits, k=1, dim=-1)
                 idx = torch.cat((idx, idx_next), dim=1)
                 continue # Skip the rest of the sampling logic

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set logits below the k-th threshold to -infinity
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        self.train() # Set model back to training mode if needed elsewhere
        return idx


class TextDataset(Dataset):
    def __init__(self, tokens_list, block_size):
        self.tokens_list = tokens_list
        self.block_size = block_size

        # Precompute all valid (doc_idx, start_pos) pairs
        self.valid_indices = []
        for doc_idx, tokens in enumerate(tokens_list):
            # For each document, find all valid starting positions
            doc_len = len(tokens)
            if doc_len >= block_size + 1:  # Need at least block_size+1 tokens
                valid_starts = doc_len - block_size
                for start_pos in range(valid_starts):
                    self.valid_indices.append((doc_idx, start_pos))

        print(f"TextDataset created with {len(tokens_list)} documents and {len(self.valid_indices)} valid sequences.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Direct lookup instead of search
        doc_idx, start_pos = self.valid_indices[idx]

        # Get document tokens
        doc_tokens = self.tokens_list[doc_idx]

        # Extract sequence and target
        end_pos = start_pos + self.block_size
        x = torch.tensor(doc_tokens[start_pos:end_pos], dtype=torch.long)
        y = torch.tensor(doc_tokens[start_pos + 1:end_pos + 1], dtype=torch.long)

        return x, y


def load_shard(shard_idx, shard_paths, metadata, pbar=None):
    shard_info = metadata['shards'][shard_idx]
    shard_docs = []

    # Memory map the shard file (near-instant)
    mmap_array = np.memmap(
        shard_paths[shard_idx],
        dtype=np.int32,
        mode='r',
        shape=(shard_info['total_tokens'],)
    )

    # Extract documents using the offsets
    doc_offsets = shard_info['doc_offsets']
    doc_lengths = shard_info['doc_lengths']

    for i in range(len(doc_lengths)):
        start = doc_offsets[i]
        length = doc_lengths[i]
        # Convert each document's tokens to a Python list
        doc_tokens = mmap_array[start:start + length].tolist()
        shard_docs.append(doc_tokens)

    # Update progress bar if provided
    if pbar is not None:
        pbar.update(1)

    return shard_docs


def load_fineweb_dataset(subset_name, sample_size, tokenizer, block_size, data_cache_dir="data_cache", num_shards=64):
    """
    Load and process a specified subset of the FineWeb dataset with optimized parallel loading.
    Uses data sharding and memory mapping for high performance on multi-core systems.
    """
    base_path = Path(data_cache_dir) / f"fineweb_{subset_name}_{sample_size}"
    base_path.parent.mkdir(exist_ok=True, parents=True)

    # Path to metadata file containing document info and vocabulary size
    metadata_path = base_path.with_suffix('.json')

    # Check if the sharded data already exists
    shard_paths = [base_path.with_suffix(f'.shard{i}.mmap') for i in range(num_shards)]
    shards_exist = all(p.exists() for p in shard_paths) and metadata_path.exists()

    if shards_exist:
        # Fast path: Load sharded data in parallel using memory mapping
        print(f"Loading sharded document data using {num_shards} parallel processes...")

        # Load metadata first (fast)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        vocab_size = metadata['vocab_size']
        num_documents = metadata['num_documents']

        # Process shards in parallel
        with tqdm(total=num_shards, desc="Loading data shards") as pbar:
            load_shard_with_args = partial(
                load_shard,
                shard_paths=shard_paths,
                metadata=metadata,
                pbar=pbar
            )

            all_docs_tokens = []
            with ThreadPoolExecutor(max_workers=min(num_shards, 64)) as executor:
                for shard_docs in executor.map(load_shard_with_args, range(num_shards)):
                    all_docs_tokens.extend(shard_docs)

            print(f"Successfully loaded {len(all_docs_tokens)} documents with {sum(len(doc) for doc in all_docs_tokens):,} total tokens.")
            return all_docs_tokens, vocab_size

    # No cache exists - we need to process the raw dataset
    print(f"Loading FineWeb subset '{subset_name}'...")
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb", name=subset_name, split="train")
        print(f"Subset '{subset_name}' loaded with {len(dataset)} documents.")
    except Exception as e:
        print(f"Error loading dataset subset '{subset_name}': {e}")
        print("Please ensure the subset name is correct (e.g., 'sample-10BT', 'sample-100BT', 'CC-MAIN-2023-50').")
        raise

    # Take the requested number of samples
    actual_samples = min(sample_size, len(dataset))
    if actual_samples < sample_size:
        print(f"Warning: Requested sample_size {sample_size} is larger than the loaded subset size {len(dataset)}. "
              f"Using all {len(dataset)} documents from the subset.")

    if actual_samples < len(dataset):
        # Sample only if needed
        print(f"Randomly sampling {actual_samples} documents from the subset...")
        import random
        indices = random.sample(range(len(dataset)), actual_samples)
        subset_view = dataset.select(indices)
        print(f"Selected {len(subset_view)} random documents for processing.")
    else:
        subset_view = dataset
        print(f"Using all {len(subset_view)} documents from the loaded subset for processing.")

    # Tokenize and process the data
    print(f"Tokenizing {len(subset_view)} documents...")
    all_docs_tokens = []

    # Determine EOS token ID for the tokenizer
    if hasattr(tokenizer, 'eot_token'):
        eos_token_id = tokenizer.eot_token
    elif hasattr(tokenizer, 'eos_token'):
        eos_token_id = tokenizer.eos_token
    else:
        eos_token_id = 100257  # cl100k_base <|endoftext|> token

    max_token_value = -1

    # Process documents
    for doc in tqdm(subset_view):
        try:
            text = doc["text"]
            if not text or text.isspace():
                continue

            doc_tokens = tokenizer.encode(text)

            if doc_tokens:
                all_docs_tokens.append(doc_tokens)
                max_in_doc = max(doc_tokens)
                if max_in_doc > max_token_value:
                    max_token_value = max_in_doc

        except Exception as e:
            print(f"Error processing document (ID: {doc.get('id', 'N/A')}): {e}")

    # Determine vocab_size
    vocab_size = tokenizer.n_vocab
    if max_token_value >= vocab_size:
        print(
            f"Warning: Max token ID found ({max_token_value}) is >= declared vocab size ({vocab_size}). Adjusting vocab_size.")
        vocab_size = max_token_value + 1

    total_tokens = sum(len(doc) for doc in all_docs_tokens)
    print(f"Created dataset with {len(all_docs_tokens)} documents and {total_tokens:,} total tokens.")

    # Save in the optimized format for future use
    print(f"Saving processed documents in optimized sharded format...")
    save_sharded_data(all_docs_tokens, vocab_size, base_path, num_shards)

    return all_docs_tokens, vocab_size


def save_sharded_data(all_docs_tokens, vocab_size, base_path, num_shards):
    """
    Save data in multiple memory-mapped shards for parallel loading.
    Also saves metadata for efficient reconstruction.
    """
    # Prepare document distribution across shards
    num_docs = len(all_docs_tokens)
    docs_per_shard = math.ceil(num_docs / num_shards)

    # Prepare metadata structure
    metadata = {
        'vocab_size': vocab_size,
        'num_documents': num_docs,
        'num_shards': num_shards,
        'shards': []
    }

    # Process each shard
    for shard_idx in tqdm(range(num_shards), desc="Creating data shards"):
        # Determine which documents go in this shard
        start_doc = shard_idx * docs_per_shard
        end_doc = min((shard_idx + 1) * docs_per_shard, num_docs)
        shard_docs = all_docs_tokens[start_doc:end_doc]

        if not shard_docs:  # Skip empty shards
            metadata['shards'].append({
                'total_tokens': 0,
                'doc_offsets': [],
                'doc_lengths': []
            })
            continue

        # Calculate total tokens and prepare arrays
        doc_lengths = [len(doc) for doc in shard_docs]
        total_tokens = sum(doc_lengths)

        # Create memory-mapped array for this shard
        shard_path = base_path.with_suffix(f'.shard{shard_idx}.mmap')
        mmap_array = np.memmap(
            shard_path,
            dtype=np.int32,
            mode='w+',
            shape=(total_tokens,)
        )

        # Populate memory-mapped array and record offsets
        doc_offsets = []
        current_offset = 0

        for i, doc in enumerate(shard_docs):
            doc_offsets.append(current_offset)
            # Convert to numpy array for efficient copying
            doc_array = np.array(doc, dtype=np.int32)
            mmap_array[current_offset:current_offset + len(doc)] = doc_array
            current_offset += len(doc)

        # Flush changes to disk
        mmap_array.flush()
        del mmap_array  # Close the memory-mapped file

        # Store shard metadata
        metadata['shards'].append({
            'total_tokens': total_tokens,
            'doc_offsets': doc_offsets,
            'doc_lengths': doc_lengths
        })

    # Save metadata
    metadata_path = base_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    print(f"Successfully saved {num_docs} documents across {num_shards} shards.")
    print(f"Metadata saved to {metadata_path}")


def get_lr_multiplier(step: int, warmup_steps: int, max_steps: int, initial_lr: float, min_lr: float) -> float:
    """Calculates the learning rate multiplier based on warmup and cosine decay."""
    if initial_lr <= 0:
        return 0.0

    min_lr_ratio = min_lr / initial_lr

    if step < warmup_steps:
        # Multiplier increases linearly from near 0 up to 1.0
        # Handle warmup_steps=0 case to avoid division by zero
        if warmup_steps == 0:
            return 1.0
        return float(step + 1) / float(warmup_steps) # step starts at 0

    # Don't decay past min rate
    if step >= max_steps:
        return min_lr_ratio

    # Cosine decay
    decay_steps = max_steps - warmup_steps
    if decay_steps <= 0:
        return min_lr_ratio
    progress = float(step - warmup_steps) / float(decay_steps)
    progress = max(0.0, min(1.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = min_lr_ratio + coeff * (1.0 - min_lr_ratio)
    return multiplier


def train_model(model, train_loader, val_loader, args, device, tokenizer): # Added tokenizer for sampling
    """Train the model with progress tracking"""
    # Setup optimizer with separate weight decay for different parameter types
    # Filter parameters for weight decay
    decay_params = []
    nodecay_params = []
    for pn, p in model.named_parameters():
        if p.requires_grad:
            # Avoid decay on bias, LayerNorm/RMSNorm weights
            if pn.endswith(".bias") or ("norm.weight" in pn):
                nodecay_params.append(p)
            else:
                decay_params.append(p)

    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    # Create AdamW optimizer and use the fused version if available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), **extra_args)
    if use_fused: print("Using fused AdamW.")

    max_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr_multiplier(
        step=step,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps,
        initial_lr=args.lr,
        min_lr=args.min_lr
    ))

    # Setup mixed precision if requested
    use_amp = args.amp and torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
    scaler = GradScaler(enabled=use_amp)
    if use_amp: print("Using Automatic Mixed Precision (AMP).")
    else: print("AMP not available or not enabled.")

    def clear_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()

    # Training loop
    best_val_loss = float('inf')
    tokens_processed = 0
    start_time = time.time()
    clear_memory()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        clear_memory()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (x, y) in pbar:
            x, y = x.to(device), y.to(device)

            # Forward pass under autocast if using AMP
            with autocast(device_type=device.type, enabled=use_amp):
                 logits, loss = model(x, y)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            # Reset gradients for the next iteration
            optimizer.zero_grad(set_to_none=True)

            # Update learning rate
            scheduler.step() # Update LR based on the current step

            # Update metrics
            loss_item = loss.item()
            epoch_loss += loss_item
            tokens_processed += x.numel()
            curr_lr = scheduler.get_last_lr()[0]
            del x, y, logits, loss

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_item:.4f}",
                'avg_loss': f"{epoch_loss / (i + 1):.4f}",
                'lr': f"{curr_lr:.2e}"
            })

            # Run validation periodically
            if args.val_interval > 0 and (i + 1) % args.val_interval == 0:
                clear_memory()
                val_loss = evaluate(model, val_loader, device)
                print(f"\nStep {i + 1}, Val loss: {val_loss:.4f}, Perplexity: {math.exp(val_loss):.2f}")
                if val_loss < best_val_loss:
                     best_val_loss = val_loss
                     print("New best validation loss.")
                     # Optional: Save best model checkpoint here
                     if args.save_dir:
                         save_path = Path(args.save_dir) / f"model_best_val.pt"
                         save_path.parent.mkdir(exist_ok=True, parents=True)
                         torch.save({
                             'epoch': epoch,
                             'step': i + 1,
                             'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'scheduler_state_dict': scheduler.state_dict(),
                             'config': model.config,
                             'val_loss': best_val_loss,
                         }, save_path)
                         print(f"Best model saved to {save_path}")

                if args.sample_during_training:
                    sample_text(model, tokenizer, device, prompt="The future of AI is", max_tokens=50)

                model.train()  # Back to training mode

        # End of epoch validation
        val_loss = evaluate(model, val_loader, device)
        avg_train_loss = epoch_loss / len(train_loader)
        end_time = time.time()
        epoch_duration = end_time - start_time
        throughput = tokens_processed / epoch_duration

        print(
            f"Epoch {epoch + 1} complete | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Perplexity: {math.exp(val_loss):.2f} | "
            f"Duration: {epoch_duration:.2f}s | "
            f"Throughput: {throughput:,.0f} tokens/sec"
        )

         # Check if current val loss is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("New best validation loss at end of epoch.")
            if args.save_dir:
                 save_path = Path(args.save_dir) / f"model_best_val.pt"
                 save_path.parent.mkdir(exist_ok=True, parents=True)
                 torch.save({
                     'epoch': epoch,
                     'step': len(train_loader), # End of epoch step
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(),
                     'config': model.config,
                     'val_loss': best_val_loss,
                 }, save_path)
                 print(f"Best model saved to {save_path}")


        # Generate sample text
        if args.sample_during_training:
            sample_text(model, tokenizer, device, prompt="The future of AI is", max_tokens=50)

        # Save checkpoint at the end of each epoch
        if args.save_dir:
            save_path = Path(args.save_dir) / f"model_epoch_{epoch + 1}.pt"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
                'val_loss': val_loss,
                'config': model.config,
            }, save_path)
            print(f"Epoch {epoch+1} model saved to {save_path}")

        # Reset for next epoch
        tokens_processed = 0
        start_time = time.time()


    return model


@torch.no_grad() # Use no_grad for evaluation efficiency
def evaluate(model, val_loader, device):
    """Evaluate model on validation data"""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    total_batches = 0

    pbar = tqdm(val_loader, desc="Evaluating", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with autocast(device_type=device.type, enabled=args.amp):
             logits, loss = model(x, y)

        if loss is not None:
             total_loss += loss.item()
             total_batches += 1
        pbar.set_postfix({'avg_loss': f"{total_loss / total_batches:.4f}"})

    model.train() # Set back to train mode
    if total_batches == 0: return float('inf') # Handle empty loader case
    return total_loss / total_batches


def sample_text(model, tokenizer, device, prompt="Once upon a time", max_tokens=100, temperature=0.8, top_k=50):
    """Generate and print a sample text"""
    print(f"\n--- Generating sample from prompt: '{prompt}' ---")

    # Tokenize the prompt
    # Make sure the tokenizer handles prepending the BOS token if required by the model
    # Example: encode(prompt, add_special_tokens=True) might be needed depending on tokenizer & model
    tokens = tokenizer.encode(prompt)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # Add batch dimension

    # Generate
    generated_tokens = model.generate(tokens_tensor, max_tokens, temperature=temperature, top_k=top_k)

    # Decode
    generated_text = tokenizer.decode(generated_tokens[0].tolist()) # Decode the first (and only) batch item
    print(f"{generated_text}")
    print("--- End of sample ---")
    return generated_text

# Need inspect for checking AdamW args
import inspect

def main():
    parser = argparse.ArgumentParser(description="Train a GPT model on FineWeb dataset")

    # Model parameters
    parser.add_argument("--block_size", type=int, default=512, help="Context window size")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--no_rope", action="store_true", help="Disable rotary positional embeddings")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing (if implemented)")

    # Dataset parameters
    # TODO consider allowing other datasets? dataset mixing?
    parser.add_argument("--fineweb_subset", type=str, default="sample-10BT",
                        help="FineWeb subset to use (e.g., 'sample-10BT', 'sample-100BT', 'CC-MAIN-2023-50')")
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Max documents to sample *from the chosen subset* for tokenization")

    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size per device")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation set fraction (from loaded documents)")
    parser.add_argument("--data_cache", type=str, default="data_cache", help="Directory to cache processed data")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader worker processes (0 for main process)")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=6e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate (after decay)")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (max norm)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Learning rate warmup steps")
    parser.add_argument("--val_interval", type=int, default=200, help="Validation interval (steps, 0 to disable)")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output parameters
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--sample_during_training", action="store_true", help="Generate sample text during training")

    # Global args variable accessible in evaluate function for AMP check
    global args
    args = parser.parse_args()


    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if torch.cuda.get_device_capability()[0] >= 9:
            print("Ampere or newer GPU detected... enabling Hopper")
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.cuda.set_per_process_memory_fraction(0.92)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    try:
        global tokenizer # Make accessible in train/sample functions
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        print("Tiktoken not installed. Please run 'pip install tiktoken'")
        exit(1)
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        exit(1)

    # --- Data Loading ---
    print("-" * 30)
    print("Loading and processing dataset...")
    # Pass the chosen subset name to the loading function
    all_docs_tokens, vocab_size = load_fineweb_dataset(
        subset_name=args.fineweb_subset,
        sample_size=args.sample_size,
        tokenizer=tokenizer,
        block_size=args.block_size,
        data_cache_dir=args.data_cache
    )
    if not all_docs_tokens:
        print("No documents were loaded or processed. Exiting.")
        exit(1)

    print(f"Dataset processed: {len(all_docs_tokens)} documents.")
    # Approximate total tokens again after processing
    total_tokens_in_memory = sum(len(doc) for doc in all_docs_tokens)
    print(f"Total tokens in memory: {total_tokens_in_memory:,}")

    # Create train/val split from the list of document tokens
    random.shuffle(all_docs_tokens) # Shuffle documents before splitting
    split_idx = int(len(all_docs_tokens) * (1 - args.val_split))
    train_docs_tokens = all_docs_tokens[:split_idx]
    val_docs_tokens = all_docs_tokens[split_idx:]

    print(f"Split into {len(train_docs_tokens)} train docs and {len(val_docs_tokens)} val docs.")

    # Create datasets and data loaders using the list of token lists
    train_dataset = TextDataset(train_docs_tokens, args.block_size)
    val_dataset = TextDataset(val_docs_tokens, args.block_size)

    # Determine num_workers based on platform
    num_workers = args.num_workers
    if num_workers > 0 and os.name == 'nt': # Windows check
        print("Warning: Setting num_workers=0 on Windows to avoid potential issues.")
        num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'), # Pin memory only if using CUDA
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else 0,
    )

    print(f"Train sequences: {len(train_dataset)}, Train batches: {len(train_loader)}")
    print(f"Val sequences: {len(val_dataset)}, Val batches: {len(val_loader)}")
    print("-" * 30)

    # --- Model Initialization ---
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_rope=not args.no_rope,
        gradient_checkpointing = False,
    )

    model = GPT(config)
    model.to(device)
    print("-" * 30)

    # --- Training ---
    print("\nStarting training...")
    model = train_model(model, train_loader, val_loader, args, device, tokenizer) # Pass tokenizer
    print("-" * 30)

    # --- Final Sampling ---
    print("\nGenerating final samples...")
    for prompt in ["The meaning of life is", "Artificial intelligence will", "Once upon a time in a land far away", "The quick brown fox"]:
        sample_text(model, tokenizer, device, prompt=prompt, max_tokens=100)
    print("-" * 30)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()