import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

# 1. Configuration
max_seq_length = 2048
dtype = None # Auto detection
load_in_4bit = True

# We use a lightweight model like Llama-3.2-1B which is fast and resource friendly
# For Indic languages, you can later swap this out for Meta-Llama-3-8B or Sarvam series
model_name = "unsloth/Llama-3.2-1B-Instruct"
dataset_path = "sample_dataset.jsonl"
output_dir = "ayurslm-lora"

# 2. Load Model and Tokenizer (2x faster downloads & inference with Unsloth)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA adapters (Efficient Fine-Tuning)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank, choosing higher rank for learning Ayurvedic concepts
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is optimized
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Prompt formatting (Ayurvedic Consultation Style)
ayur_prompt = """Namaste. You are an expert Ayurvedic assistant. Below is an instruction describing a patient's symptoms or questions, followed by an Ayurvedic context or response.
Write a culturally-aware, balanced Ayurvedic response. Focus on Vata, Pitta, and Kapha doshas. Avoid modern medical diagnosis.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = ayur_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 5. Load Dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 6. Training setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Increase for actual full-training
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to="none"
    ),
)

# 7. Train Model
print("Starting training AyurSLM...")
trainer_stats = trainer.train()

# 8. Save the LoRA adapters
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
