# Transformer
Let's write our own Transformer model from scratch using PyTorch.

## GPT2
GPT-2 is a family of models released by OpenAI in 2019. At the smallest end, there is a **124M** parameter model, 
and at the largest end, there is a **1.558B** parameter model. These models are trained on a large corpus of text 
data and can generate human-like text.

In this repository, we are going to try to replace the **124M** parameter model, which has:
- 12 Transformer blocks
- 768 Dimensions (the size of a vector representation of each token)

This repository is modelled after the [hugging face](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
take on gpt-2, which is written in PyTorch instead of the original TensorFlow. 

## Prerequisites
* Python 3.8, 3.9, 3.10, 3.11, 3.12
  * https://www.python.org/downloads/
* PyTorch 
  * https://pytorch.org/get-started/locally/
  * run `nvidia-smi` to figure out which CUDA version your card supports
* Shakespeare's texts
  * You can use any text you want, but Shakespeare's texts are included in ./data/shakespeare.txt

## Installation
1. Fork the repository
2. Clone the repository
  ```bash
  git clone https://github.com/<your-username>/transformer.git
  ```
3. Open `train_gpt2.py`

You will need `tiktoken` for tokenizing the text. You can install it using pip:
```bash
pip install tiktoken
```

## Credits

* First hour of: https://youtu.be/l8pRSuU81PU
* Intuition behind variable names: https://youtu.be/eMlx5fFNoYc
* Intuition behind variable names: https://youtu.be/9-Jl0dxWQs8