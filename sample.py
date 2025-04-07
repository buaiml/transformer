import argparse
import torch
import tiktoken
from pathlib import Path
import sys
import json

# Import the necessary model classes
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    use_rope: bool = True
    gradient_checkpointing: bool = False


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]  # Shape: (1, 1, seq_len, dim)
            self.sin_cached = emb.sin()[None, None, :, :]  # Shape: (1, 1, seq_len, dim)
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head

        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd)
        self.dropout = torch.nn.Dropout(config.dropout)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size), persistent=False)

        self.rotary_emb = RotaryEmbedding(self.head_size) if config.use_rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(q, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.dropout(y)
        return y


class MLP(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = ((n_hidden + 256 - 1) // 256) * 256

        self.w1 = torch.nn.Linear(config.n_embd, n_hidden, bias=False)
        self.w3 = torch.nn.Linear(config.n_embd, n_hidden, bias=False)
        self.w2 = torch.nn.Linear(n_hidden, config.n_embd, bias=False)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x)
        return x


class Block(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attn(self.ln_1(x))
        x = x + attn_output
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output
        return x


class GPT(torch.nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = torch.nn.ModuleDict(dict(
            token_embeddings=torch.nn.Embedding(config.vocab_size, config.n_embd),
            positional_embeddings=torch.nn.Embedding(config.block_size, config.n_embd) if not config.use_rope else None,
            dropout=torch.nn.Dropout(config.dropout),
            blocks=torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=torch.nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights
        self.transformer.token_embeddings.weight = self.lm_head.weight

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        B, T = tokens.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        x = self.transformer.token_embeddings(tokens)

        if not self.config.use_rope and self.transformer.positional_embeddings is not None:
            positions = torch.arange(0, T, dtype=torch.long, device=tokens.device)
            pos_emb = self.transformer.positional_embeddings(positions)
            x = x + pos_emb

        x = self.transformer.dropout(x)

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                                     ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, callback=None):
        """Generate text with controlled sampling"""
        self.eval()
        for i in range(max_new_tokens):
            # Get the sequence so far
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            else:
                # Greedy sampling
                _, idx_next = torch.topk(logits, k=1, dim=-1)
                idx = torch.cat((idx, idx_next), dim=1)
                if callback: callback(idx)
                continue

            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample from the distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token
            idx = torch.cat((idx, idx_next), dim=1)

            # Call the callback function if provided
            if callback: callback(idx)

        return idx


def load_model(checkpoint_path, device):
    """Load the model from a checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Try to reconstruct config (might need adjustments based on your checkpoint format)
        config = GPTConfig(
            block_size=512,  # Default
            vocab_size=100277,  # Default for cl100k_base
            n_layer=6,  # Default
            n_head=8,  # Default
            n_embd=512,  # Default
            dropout=0.1,  # Set to 0 for inference
            use_rope=True,  # Default
            gradient_checkpointing=False  # Not needed for inference
        )

    # Create model with the config
    model = GPT(config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the entire model was saved directly
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # Set to evaluation mode
    return model, config


def setup_tokenizer():
    """Setup the tokenizer"""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except:
        print("Error: Could not load the cl100k_base tokenizer.")
        print("Please ensure tiktoken is installed: pip install tiktoken")
        sys.exit(1)


def print_completion_stats(prompt, completion, time_taken=None):
    """Print statistics about the completion"""
    print("\n" + "-" * 50)
    print(f"Tokens in prompt: {len(prompt)}")
    print(f"Tokens in completion: {len(completion) - len(prompt)}")
    if time_taken:
        print(f"Generation time: {time_taken:.2f} seconds")
        tokens_per_sec = (len(completion) - len(prompt)) / time_taken
        print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
    print("-" * 50 + "\n")


def interactive_mode(model, tokenizer, device, max_tokens, temperature, top_k, top_p, stream):
    """Run the model in interactive mode with user input"""
    print("GPT Interactive Mode - Enter prompts to get completions")
    print("Type 'exit', 'quit', or press Ctrl+C to exit")
    print("Type 'settings' to view/change generation parameters")
    print("-" * 50)

    settings = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stream": stream
    }

    while True:
        try:
            prompt = input("\nPrompt> ")

            if prompt.lower() in ['exit', 'quit']:
                break

            if prompt.lower() == 'settings':
                print("\nCurrent settings:")
                for k, v in settings.items():
                    print(f"{k}: {v}")

                change = input("\nChange settings? (y/n) ")
                if change.lower() == 'y':
                    for setting in settings:
                        new_val = input(f"New value for {setting} [{settings[setting]}]: ")
                        if new_val:
                            try:
                                if setting == "stream":
                                    settings[setting] = new_val.lower() in ['true', 'yes', 'y', '1']
                                else:
                                    settings[setting] = type(settings[setting])(new_val)
                            except ValueError:
                                print(f"Invalid value for {setting}, keeping current value")

                print("\nUpdated settings:")
                for k, v in settings.items():
                    print(f"{k}: {v}")
                continue

            if not prompt:
                continue

            tokens = tokenizer.encode(prompt)
            input_tensor = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]

            import time
            start_time = time.time()

            if settings["stream"]:
                print("\nGenerating: ", end="", flush=True)
                generated_tokens = []

                def callback(idx):
                    # Get the newly generated token
                    new_token = idx[0, -1].item()
                    generated_tokens.append(new_token)
                    token_str = tokenizer.decode([new_token])
                    print(token_str, end="", flush=True)

                model.generate(
                    input_tensor,
                    max_new_tokens=settings["max_tokens"],
                    temperature=settings["temperature"],
                    top_k=settings["top_k"] if settings["top_k"] > 0 else None,
                    top_p=settings["top_p"] if settings["top_p"] > 0 else None,
                    callback=callback
                )
                print()  # Add newline after streaming
                completion = tokens + generated_tokens
            else:
                output = model.generate(
                    input_tensor,
                    max_new_tokens=settings["max_tokens"],
                    temperature=settings["temperature"],
                    top_k=settings["top_k"] if settings["top_k"] > 0 else None,
                    top_p=settings["top_p"] if settings["top_p"] > 0 else None
                )
                completion = output[0].tolist()
                result_text = tokenizer.decode(completion)
                print(f"\nCompletion:\n{result_text}")

            time_taken = time.time() - start_time
            print_completion_stats(tokens, completion, time_taken)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def single_completion(model, tokenizer, device, prompt, max_tokens, temperature, top_k, top_p, stream):
    """Generate a single completion from a prompt"""
    tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]

    import time
    start_time = time.time()

    if stream:
        print("Generating: ", end="", flush=True)
        generated_tokens = []

        def callback(idx):
            new_token = idx[0, -1].item()
            generated_tokens.append(new_token)
            token_str = tokenizer.decode([new_token])
            print(token_str, end="", flush=True)

        model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p > 0 else None,
            callback=callback
        )
        print()  # Add newline after streaming
        completion = tokens + generated_tokens
        time_taken = time.time() - start_time
        print_completion_stats(tokens, completion, time_taken)
    else:
        output = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p > 0 else None
        )
        completion = output[0].tolist()
        result_text = tokenizer.decode(completion)
        time_taken = time.time() - start_time
        print(result_text)
        print_completion_stats(tokens, completion, time_taken)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--prompt", type=str, help="The prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (0 for greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter (0 to disable)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter (0 to disable)")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they're generated")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU.")
    else:
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file {checkpoint_path} does not exist.")
        sys.exit(1)

    print(f"Loading model from {checkpoint_path}...")
    model, config = load_model(checkpoint_path, device)
    print(f"Model loaded! Config: {config}")

    tokenizer = setup_tokenizer()

    if args.interactive:
        interactive_mode(
            model, tokenizer, device,
            args.max_tokens, args.temperature, args.top_k, args.top_p, args.stream
        )
    elif args.prompt:
        single_completion(
            model, tokenizer, device, args.prompt,
            args.max_tokens, args.temperature, args.top_k, args.top_p, args.stream
        )
    else:
        print("Error: Either --prompt or --interactive must be specified.")
        sys.exit(1)


if __name__ == "__main__":
    main()
