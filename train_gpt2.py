from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MultiHeadAttention(nn.Module):
    """
    The multi-head attention mechanism allows vectors to "ask" queries to vectors that came
    before them in the sequence. This should allow the model to learn dependencies between words in a sentence.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MLP(nn.Module):
    """
    The multi-layer perceptron is responsible for taking a vector (a representation
    of a word) and looking up "facts" about that vector.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.up_projection = nn.Linear(config.n_embd, 4 * config.n_embd)  # Up projection to a larger space
        self.gelu = nn.GELU()  # Activation function
        self.down_projection = nn.Linear(4 * config.n_embd, config.n_embd)  # Down projection back to normal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class Block(nn.Module):
    """
    A single block of the GPT model, which consists of a multi-head self-attention layer
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.n_embd)
        self.attention = MultiHeadAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.n_embd)
        self.multi_layer_perceptron = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class GPT(nn.Module):
    """
    Responsible for taking a list of tokens (integers) and passing them through
    the transformer model, then outputting new tokens.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            token_embeddings = nn.Embedding(config.vocab_size, config.n_embd),
            positional_embeddings = nn.Embedding(config.block_size, config.n_embd),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            layer_norm = nn.LayerNorm(config.n_embd),
        ))
        self.token_decoder = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass



# Actually doing something with our model
if __name__ == "__main__":
    config = GPTConfig()
    num_return_sequences = 5
    max_length = 50  # Maximum length of the generated sequence

    # Try to grab the fastest device
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # For MacOS with Apple Silicon
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # tokens
    import tiktoken
    encoding = tiktoken.get_encoding("gpt2")
    tokens = encoding.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    # Create the model
    model = GPT(config)
    model.to(device)

    # TODO: implement training data

    # generate
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    # print generated response
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = encoding.decode(tokens)
        print(">", decoded)
