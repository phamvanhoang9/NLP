# LoRA

LoRA (Low-Rank Adaptation of Large Language Models) is a popular and lightweight training technique that 
significantly reduces the number of trainable parameters. It works by inserting a smaller number of new weights into the model and only these are trained. This makes training with LoRA much faster, memory-efficient, and produces smaller model weights (a few hundred MBs), which are easier to store and share. LoRA can also be combined with other training techniques like DreamBooth to speedup training.

Explanation of LoRA: [here](https://youtu.be/lixMONUAjfs?si=ZvefgYhdWaEX63_6).

# How to use LoRA with Huggingface Transformers

LoRA can be easily used with Huggingface Transformers. Here is a step-by-step guide to use LoRA with Huggingface Transformers:

1. Install the required libraries:
```bash
pip install torch transformers
```

2. Load the pre-trained model:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
```

3. Define the LoRA model:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class LoRAModel(torch.nn.Module):
    def __init__(self, model, tokenizer, rank=32):
        super(LoRAModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        self.lora_weights = torch.nn.Parameter(torch.randn(self.rank, self.model.config.n_embd))
        self.lora_bias = torch.nn.Parameter(torch.randn(self.rank))

    def forward(self, input_ids, **kwargs):
        hidden_states = self.model(input_ids, **kwargs).last_hidden_state
        lora_weights = torch.nn.functional.linear(self.lora_weights, self.model.transformer.wte.weight)
        lora_bias = torch.nn.functional.linear(self.lora_bias, self.model.transformer.wte.weight)
        lora_logits = torch.einsum("bld,ld->bl", hidden_states, lora_weights) + lora_bias
        return lora_logits
```

4. Train the LoRA model:
```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

lora_model = LoRAModel(model, tokenizer, rank=32)
lora_model.train()

optimizer = torch.optim.Adam(lora_model.parameters(), lr=1e-4)
for i in range(1000):
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids # Example input
    logits = lora_model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits, input_ids[:, 1:])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

5. Save the LoRA model:
```python
lora_model.save_pretrained("lora_model")
```

# References

Fine-tune FLAN-T5 for chat & dialogue summarization with LoRA: [here](https://www.philschmid.de/fine-tune-flan-t5).
