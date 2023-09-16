import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


print(f"Notebook runtime: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"PyTorch version : {torch.__version__}")
print(f"Transformers version : {datasets.__version__}")
print(f"Datasets version : {transformers.__version__}")

# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Dataset
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
#compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Load model in 4-bit precision. Alternatively, you can download it from HF already quantized
    bnb_4bit_quant_type="nf4", # Quantization type (fp4 or nf4)
    bnb_4bit_compute_dtype=torch.float16, # OJO: sin comillas. Compute dtype for 4-bit base models
    bnb_4bit_use_double_quant=False, # Activate nested quantization for 4-bit base models (double quantization)
)

# Load the entire model on the GPU 0
device_map = {"": 0}

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1


# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16, # factor de escalado alpha/r de los pesos de las nuevas matrices. Si alto, más perso a las activaciones
    lora_dropout=0.1, # Dropout probability for LoRA layers
    r=64, # LoRA attention dimension
    bias="none",
    task_type="CAUSAL_LM",
)


# Set SFT training parameters
training_arguments = TrainingArguments(
    output_dir='/qlora_model',
    num_train_epochs=1, # Number of training epochs
    per_device_train_batch_size=4, # Batch size per GPU for training
    gradient_accumulation_steps=1, # Number of update steps to accumulate the gradients for
    optim="paged_adamw_32bit", # Optimizer to use
    save_steps=0,
    logging_steps=25, # Log every X updates steps
    learning_rate=2e-4, # Initial learning rate (AdamW optimizer)
    weight_decay=0.001, # Weight decay to apply to all layers except bias/LayerNorm weights
    fp16=False, # Enable fp16/bf16 training (set bf16 to True with an A100)
    bf16=False, # Enable fp16/bf16 training (set bf16 to True with an A100)
    max_grad_norm=0.3, # Evita el explosion_gradient. Maximum gradient normal (gradient clipping)
    max_steps=-1, # Number of training steps (overrides num_train_epochs)
    warmup_ratio=0.03, # Ratio of steps for a linear warmup (from 0 to learning rate)
    group_by_length=True, # Group sequences into batches with same length. Saves memory and speeds up training considerably
    lr_scheduler_type= "cosine", # Learning rate schedule
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None, ## Maximum sequence length to use
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False, # Pack multiple short examples in the same input sequence to increase efficiency
)

# Train model
trainer.train()

# Save trained model
new_model = "mi_modelo_qlora"
trainer.model.save_pretrained(new_model)

# Ignore warnings

## MERGE FORMER MODEL WITH loRA weights

# Empty VRAM
del model
del pipe
del trainer
import gc
gc.collect()
gc.collect()

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, "/content/mi_modelo_qlora")
model = model.merge_and_unload()

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])


## missing upload model to Model registry
