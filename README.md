# Transformer
Let's train our transformer! Here we have a complete transformer implementation
with some sneaky optimizations to make training easier (since we are paying for
GPUs, we should be maximizing our value!). 

## Lambda Labs
We will be training on either H100 or H200 GPU(s) (depending on what your budget is).
Before we can start, you will need to:
1. Create a LambdaLabs account
2. Set up your billing information
3. Create an SSH key and save it

On a unix system, you should move your SSH key to `~/.ssh/buais.peb`. On a windows
system, you should move your SSH key to `C:\Users\<username>\.ssh\buais.peb`.
```
# For MacOS/Linux
mv /path/to/your/key ~/.ssh/buais.peb
chmod 600 ~/.ssh/buais.peb
```

#### Connecting to Lambda Labs
Once you have your SSH key, you should [create an instance](https://cloud.lambda.ai/instances) by
clicking the "Launch instance" button in the top right. This repository is set up
for you to use the `1x GH200 (96 GB)` instance, since it is very good, especially
for its price. Then:
1. Select the nearest available region
2. Click "Don't attach a filesystem" (we aren't working with datasets above 1TB)
3. Click "Launch instance"

This will take a few moments... Once status says `Running`, you can copy the
SSH Login command (should look something like `ssh ubuntu@<ip address>`).

Then, run the following
```bash
# For MacOS/Linux
ssh -i ~/.ssh/buais.peb ubuntu@<ip address>

# For Windows Git Bash
ssh -i /c/Users/<username>/.ssh/buais.peb ubuntu@<ip address>
```

> **Pro Tip**: You can create multiple terminals by running the same command in new terminals.

## Repository instructions
First, we should clone the repository:
```bash
git clone https://github.com/buaiml/transformer.git
cd transformer
git checkout training
tree
```

The repository relies on `tqdm` to show progress bars, `datasets` to load huggingface
datasets, and `tiktoken` to tokenize the data. You can install these with:
```bash
pip install tqdm datasets tiktoken
```

Then you are all set! :)

## Training
To train a model, you can run the following command:

```bash
python train_gpt2 \
--block_size 128 \  # Use something small here so training goes fast
--n_layer 12 \
--n_head 12 \
--n_embd 768 \
--lr 8e-4 \
--min_lr 8e-5 \
--batch_size 128 \ 
--val_split 0.05 \  # 5% of the data will be used for validation
--num_workers 4 \
--warmup_steps 500 \
--amp \
--dropout 0.2 \
--sample_size 1000
```

This will get you started with downloading a subset of the fineweb dataset, and
start training! But we might run into some issues during training, so we might
need to tweak our parameters!

## Evaluation
While the model is training, we can still test out snapshots!
```bash
python sample.py \
--interactive \
--stream \
--checkpoint ./checkpoints/model_best_val.pt \
--cpu \  # Use the CPU so we can save the GPU for training
--max_tokens 50
```
This will start a simple text-generation loop for you to play with... Try
out some silly prompts for the model to complete.

## Saving
Once you have a model you like, you can download it to your local machine!
Make sure you stop training first, then run the following command:
```bash
# For MacOS/Linux
scp -i ~/.ssh/buais.peb ubuntu@<ip address>:~/transformer/checkpoints/model_best_val.pt ./checkpoints/

# For Windows Git Bash
scp -i /c/Users/<username>/.ssh/buais.peb ubuntu@<ip address>:~/transformer/checkpoints/model_best_val.pt ./checkpoints/
```

You can then run the same commands on your local machine (assuming you have this 
transformer repository cloned).

