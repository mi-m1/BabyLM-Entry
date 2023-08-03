import json

from transformers import GPT2Config, PreTrainedTokenizerFast, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max-positions", type=int)
parser.add_argument("--embed-dimension", type=int)
parser.add_argument("--layers", type=int)
parser.add_argument("--attention-heads", type=int)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--lr", type=float)
parser.add_argument("--bsz", type=int)
parser.add_argument("--warmup-steps", type=int)
parser.add_argument("--max-steps", type=int)
parser.add_argument("--eval-steps", type=int)
parser.add_argument("--save-steps", type=int)
args = parser.parse_args()

block_size = args.max_positions
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]])


# Load dataset
dataset = load_dataset("text", data_files={"train": ["../data/babylm_data/babylm_10M/train_full"], "dev": "../data/babylm_data/babylm_dev/dev_10k"})

# Load model
config = GPT2Config(
    vocab_size=16000,
    n_positions=args.max_positions,
    n_embd=args.embed_dimension,
    n_layer=args.layers,
    n_head=args.attention_heads,
    activation_function="gelu"
)
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../process-data/tokeniser-unigram-16000.json")
tokenizer.eos_token = '<|endoftext|>'
tokenizer.bos_token = '<|endoftext|>'
tokenizer.unk_token = '<|endoftext|>'
model = AutoModelForCausalLM.from_config(config)

# Tokenize data
tokenized_ds = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names,
)


lm_dataset = tokenized_ds.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    learning_rate=args.lr,
    weight_decay=0.01,
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    warmup_steps=args.warmup_steps, 
    max_steps=args.max_steps,
    per_device_train_batch_size=args.bsz,
    gradient_accumulation_steps=1,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["dev"],
    data_collator=data_collator,
)

trainer.train()
