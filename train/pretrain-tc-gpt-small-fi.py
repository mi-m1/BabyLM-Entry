import json
from transformers import GPT2Config, PreTrainedTokenizerFast, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import argparse
import glob

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
parser.add_argument("--tc-scheme", type=str)
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
    # return tokenizer([" ".join(x) for x in examples["text"]])
    return tokenizer([" ".join(examples["text"])])

# Define the curriculum datasets
datasets = {
    "1": {
        "train": [f"../data-{args.tc_scheme}/curriculum1.train"],
        "dev": f"../data-{args.tc_scheme}/curriculum1.dev"
    },
    "2": {
        "train": [f"../data-{args.tc_scheme}/curriculum2.train"],
        "dev": f"../data-{args.tc_scheme}/curriculum2.dev"
    },
    "3": {
        "train": [f"../data-{args.tc_scheme}/curriculum3.train"],
        "dev": f"../data-{args.tc_scheme}/curriculum3.dev"
    },
    "4": {
        "train": [f"../data-{args.tc_scheme}/curriculum4.train"],
        "dev": f"../data-{args.tc_scheme}/curriculum4.dev"
    },
    "5": {
        "train": [f"../data-{args.tc_scheme}/curriculum5.train"],
        "dev": f"../data-{args.tc_scheme}/curriculum5.dev"
    },
    "6": {
        "train": [f"../data-{args.tc_scheme}/curriculum6.train"],
        "dev": f"../data-{args.tc_scheme}/curriculum6.dev"
    },
    "7": {
        "train": [f"../data-{args.tc_scheme}/curriculum7.train"],
        "dev": f"../data-{args.tc_scheme}/curriculum7.dev"
    },
    "8": {
        "train": [f"../data-{args.tc_scheme}/curriculum8.train"],
        "dev": f"../data-{args.tc_scheme}/curriculum8.dev"
    },
    "9": {
        "train": [f"../data-{args.tc_scheme}/curriculum9.train"],
        "dev": f"../data-{args.tc_scheme}/curriculum9.dev"
    },
    "10": {
        "train": [f"../data-{args.tc_scheme}/curriculum10.train"],
        "dev": f"../data-{args.tc_scheme}/curriculum10.dev"
    }
}

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

# Perform curriculum learning
for level, dataset_files in datasets.items():
    # Load dataset for the current level
    dataset = load_dataset("text", data_files={"train": dataset_files["train"], "dev": dataset_files["dev"]})
    # print("dataset")
    # print(dataset["train"][0])

    # Tokenize data
    tokenized_ds = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )
    
    # print("tokenised dataset:")
    # print(tokenized_ds[0])
    # print(tokenized_ds["train"][0])


    lm_dataset = tokenized_ds.map(group_texts, batched=True, num_proc=4)
    # print(lm_dataset["train"][0])
    # print(lm_dataset["train"])
    
    #number of batches
    num_batches = lm_dataset["train"].num_rows
    print(f"This is the number of batches in curriculum{level}: {num_batches}")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train
    training_args = TrainingArguments(
        # output_dir=args.output_dir + "/level_" + level,
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        # save_steps=args.save_steps,
        save_steps = num_batches,
        # eval_steps=args.eval_steps,
        eval_steps = num_batches,
        warmup_steps = num_batches * 0.06, 
        # max_steps=args.max_steps,
        max_steps = num_batches, # times 2 to see the whole dataset twice
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=1,
        logging_steps=100, # for output file
    )

    model = AutoModelForCausalLM.from_config(config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["dev"],
        data_collator=data_collator,
    )

    if level == "1":
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=True)

    # break
    
