import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import gc
from collections import defaultdict
import numpy as np
import re

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Function to check GPU memory usage
def print_gpu_memory(step_name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nMemory Usage at {step_name}:")
        print(f"  Allocated: {allocated:.2f} GiB")
        print(f"  Reserved: {reserved:.2f} GiB")
        print(f"  Free: {total - allocated:.2f} GiB")
        print(f"  Total: {total:.2f} GiB")

# Clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_memory("After Clearing GPU Memory")

clear_gpu_memory()

# 1. Prepare the dataset
data = "./datasetBalanced1.csv"
try:
    df = pd.read_csv(data)
except FileNotFoundError:
    print(f"Error: {data} not found.")
    raise
print("Dataset Head:")
print(df.head())
print("\nColumns in dataset:", df.columns.tolist())
print("\nNumber of unique subcategories:", df['Subcategory'].nunique())
print("Number of unique actions:", df['Action'].nunique())
print("\nSubcategory counts:")
print(df['Subcategory'].value_counts())
print("\nAction counts:")
print(df['Action'].value_counts())

# Verify required columns
required_columns = ['Category', 'Subcategory', 'Action', 'Sentence', 'Response']
if not all(col in df.columns for col in required_columns):
    print(f"Error: Dataset must contain {required_columns}. Found: {df.columns.tolist()}")
    raise ValueError("Missing required columns.")

# Normalize category names (e.g., 'Camera' and 'camera' to 'camera')
df['Category'] = df['Category'].str.lower()
df['Subcategory'] = df['Subcategory'].str.lower()

# Filter out combinations with fewer than 2 occurrences
df['stratify_key'] = df['Category'] + '_' + df['Subcategory'] + '_' + df['Action']
combination_counts = df['stratify_key'].value_counts()
valid_combinations = combination_counts[combination_counts >= 2].index
df = df[df['stratify_key'].isin(valid_combinations)]
print(f"\nAfter filtering, dataset size: {len(df)}")
print("Remaining combination counts:")
print(df['stratify_key'].value_counts())

# Create input prompt combining Category, Subcategory, Action, and Sentence
df['input'] = df.apply(lambda row: f"Category: {row['Category']}, Subcategory: {row['Subcategory']}, Action: {row['Action']}, Sentence: {row['Sentence']}", axis=1)
df['output'] = df['Response']

# Split into train and test sets
try:
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df['stratify_key']
    )
except ValueError as e:
    print(f"Stratification failed: {e}")
    print("Falling back to non-stratified split.")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df[['input', 'output', 'Category', 'Subcategory', 'Action']])
test_dataset = Dataset.from_pandas(test_df[['input', 'output', 'Category', 'Subcategory', 'Action']])

# 2. Load model and tokenizer
model_name = "./Llama-3.2-3B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()
    print_gpu_memory("After Loading Base Model")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# 3. Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print_gpu_memory("After Applying LoRA")

# 4. Preprocess dataset
def preprocess_function(examples):
    inputs = [f"Instruction: {inp}\nAssistant: {out} <|END|>" for inp, out in zip(examples['input'], examples['output'])]
    tokenized = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['input', 'output', 'Category', 'Subcategory', 'Action'])
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_llama",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=7,  # Increased epochs for better training
    learning_rate=1e-4,  # Slightly lower learning rate
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    gradient_checkpointing=True,
    optim="adamw_8bit",
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 7. Fine-tune the model
try:
    trainer.train()
    print_gpu_memory("After Training")
except torch.cuda.OutOfMemoryError as e:
    print(f"CUDA OOM Error during training: {e}")
    clear_gpu_memory()
    raise

# 8. Save the fine-tuned model
model.save_pretrained("./finetuned_llama")
tokenizer.save_pretrained("./finetuned_llama")

# 9. Evaluate test accuracy per category, subcategory, and action
model.eval()
bleu_scores = defaultdict(list)
rouge_scores = defaultdict(list)
exact_matches = defaultdict(list)
rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoother = SmoothingFunction().method1

def clean_response(text):
    """Clean generated response to remove extraneous text."""
    # Remove anything after <|END|> or other metadata
    text = text.split("<|END|>")[0].strip()
    # Remove patterns like 'Month, Day, Year', 'Assistant.', etc.
    text = re.sub(r'(Month|Day|Year|Hour|Minute|Second|Weekday|Temperature|Humidity|.*\bAssistant\b.*$)', '', text, flags=re.IGNORECASE)
    # Remove extra punctuation, spaces, and trailing text
    text = re.sub(r'\s+', ' ', text).strip('.,:; ')
    return text

for example in test_dataset:
    input_text = example['input']
    expected_response = example['output']
    category = example['Category']
    subcategory = example['Subcategory']
    action = example['Action']
    prompt = f"Instruction: {input_text}\nAssistant: "
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,  # Reduced to enforce concise responses
            num_beams=5,
            no_repeat_ngram_size=3,
            do_sample=False,
        )
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
        generated_response = clean_response(generated_response)
        bleu_score = sentence_bleu([expected_response.split()], generated_response.split(), smoothing_function=smoother)
        rouge_score = rouge_scorer.score(expected_response, generated_response)['rougeL'].fmeasure
        exact_match = 1 if generated_response.lower() == expected_response.lower() else 0
        bleu_scores[('Category', category)].append(bleu_score)
        bleu_scores[('Subcategory', subcategory)].append(bleu_score)
        bleu_scores[('Action', action)].append(bleu_score)
        rouge_scores[('Category', category)].append(rouge_score)
        rouge_scores[('Subcategory', subcategory)].append(rouge_score)
        rouge_scores[('Action', action)].append(rouge_score)
        exact_matches[('Category', category)].append(exact_match)
        exact_matches[('Subcategory', subcategory)].append(exact_match)
        exact_matches[('Action', action)].append(exact_match)
        print(f"Input: {input_text}")
        print(f"Expected: {expected_response}")
        print(f"Generated: {generated_response}")
        print(f"BLEU Score: {bleu_score:.4f}")
        print(f"ROUGE-L Score: {rouge_score:.4f}")
        print(f"Exact Match: {exact_match}\n")
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error during generation: {e}")
        clear_gpu_memory()
        continue

# 10. Calculate and display average scores per category, subcategory, and action
print("\nEvaluation Metrics by Category, Subcategory, and Action:")
for key, scores in sorted(bleu_scores.items()):
    group_type, group_name = key
    avg_bleu = np.mean(scores)
    avg_rouge = np.mean(rouge_scores[key])
    avg_exact = np.mean(exact_matches[key])
    count = len(scores)
    print(f"\n{group_type}: {group_name} (Count: {count})")
    print(f"  Average BLEU Score: {avg_bleu:.4f}")
    print(f"  Average ROUGE-L Score: {avg_rouge:.4f}")
    print(f"  Exact Match Accuracy: {avg_exact:.4f}")

# 11. Overall metrics
avg_bleu = np.mean([score for scores in bleu_scores.values() for score in scores])
avg_rouge = np.mean([score for scores in rouge_scores.values() for score in scores])
avg_exact = np.mean([match for matches in exact_matches.values() for match in matches])
print(f"\nOverall Average BLEU Score: {avg_bleu:.4f}")
print(f"Overall Average ROUGE-L Score: {avg_rouge:.4f}")
print(f"Overall Exact Match Accuracy: {avg_exact:.4f}")

# Clean up
clear_gpu_memory()

# import os
# import pandas as pd
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# from peft import LoraConfig, get_peft_model
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# import gc
# from rouge_score import rouge_scorer

# # Set environment variable to reduce memory fragmentation
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # Function to check GPU memory usage
# def print_gpu_memory(step_name):
#     torch.cuda.synchronize()
#     allocated = torch.cuda.memory_allocated(0) / 1e9
#     reserved = torch.cuda.memory_reserved(0) / 1e9
#     total = torch.cuda.get_device_properties(0).total_memory / 1e9
#     print(f"\nMemory Usage at {step_name}:")
#     print(f"  Allocated: {allocated:.2f} GiB")
#     print(f"  Reserved: {reserved:.2f} GiB")
#     print(f"  Free: {total - allocated:.2f} GiB")
#     print(f"  Total: {total:.2f} GiB")

# # Clear GPU memory
# def clear_gpu_memory():
#     torch.cuda.empty_cache()
#     gc.collect()
#     print_gpu_memory("After Clearing GPU Memory")

# clear_gpu_memory()

# # 1. Prepare the dataset
# data = "./datasetBalanced1.csv"
# try:
#     df = pd.read_csv(data)
# except FileNotFoundError:
#     print(f"Error: {data} not found.")
#     raise
# print("Dataset Head:")
# print(df.head())
# print("\nColumns in dataset:", df.columns.tolist())
# print("\nNumber of unique subcategories:", df['Subcategory'].nunique())
# print("Number of unique actions:", df['Action'].nunique())
# print("\nSubcategory counts:")
# print(df['Subcategory'].value_counts())
# print("\nAction counts:")
# print(df['Action'].value_counts())

# # Verify required columns
# required_columns = ['Category', 'Subcategory', 'Action', 'Sentence', 'Response']
# if not all(col in df.columns for col in required_columns):
#     print(f"Error: Dataset must contain {required_columns}. Found: {df.columns.tolist()}")
#     raise ValueError("Missing required columns.")

# # Create input prompt combining Category, Subcategory, Action, and Sentence
# df['input'] = df.apply(lambda row: f"Category: {row['Category']}, Subcategory: {row['Subcategory']}, Action: {row['Action']}, Sentence: {row['Sentence']}", axis=1)
# df['output'] = df['Response']

# # Split into train and test sets
# train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
# train_dataset = Dataset.from_pandas(train_df[['input', 'output']])
# test_dataset = Dataset.from_pandas(test_df[['input', 'output']])

# # 2. Load model and tokenizer
# model_name = "./Llama-3.2-3B"  
# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
#     model.gradient_checkpointing_enable()  # Save memory
#     print_gpu_memory("After Loading Base Model")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     raise

# # Set padding token
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.eos_token_id

# # 3. Apply LoRA
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
# model = get_peft_model(model, lora_config)
# print_gpu_memory("After Applying LoRA")

# # 4. Preprocess dataset
# def preprocess_function(examples):
#     inputs = [f"You are a smart assistant. Respond concisely to the user's instruction without adding metadata or follow-up questions. User: {inp} ### Assistant: {out}" for inp, out in zip(examples['input'], examples['output'])]
#     tokenized = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
#     tokenized['labels'] = tokenized['input_ids'].copy()
#     return tokenized

# train_dataset = train_dataset.map(preprocess_function, batched=True)
# test_dataset = test_dataset.map(preprocess_function, batched=True)

# # 5. Training arguments
# training_args = TrainingArguments(
#     output_dir="./finetuned_llama",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=2,
#     num_train_epochs=5,
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=10,
#     save_strategy="epoch",
#     eval_strategy="epoch",
#     load_best_model_at_end=True,
#     lr_scheduler_type="cosine",
#     warmup_steps=100,
#     gradient_checkpointing=True,
# )

# # 6. Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )

# # 7. Fine-tune the model
# try:
#     trainer.train()
#     print_gpu_memory("After Training")
# except torch.cuda.OutOfMemoryError as e:
#     print(f"CUDA OOM Error during training: {e}")
#     clear_gpu_memory()
#     raise

# # 8. Save the fine-tuned model
# model.save_pretrained("./finetuned_llama")
# tokenizer.save_pretrained("./finetuned_llama")

# # 9. Evaluate test accuracy
# model.eval()
# bleu_scores = []
# rouge_scores=[]
# rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# smoother = SmoothingFunction().method1

# for example in test_dataset:
#     input_text = example['input']
#     expected_response = example['output']
#     prompt = f"You are a smart assistant. Respond concisely to the user's instruction without adding metadata or follow-up questions. User: {input_text} ### Assistant: "
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=30,  # Reduced for concise responses
#             num_beams=5,
#             no_repeat_ngram_size=3,
#             do_sample=False,  # Disable sampling for deterministic output
#         )
#         generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Assistant:")[-1].strip()
#         # Post-process to remove anything after ###
#         generated_response = generated_response.split("###")[0].strip()
#         bleu_score = sentence_bleu([expected_response.split()], generated_response.split(), smoothing_function=smoother)
#         rouge_score = rouge_scorer.score(expected_response, generated_response)['rougeL'].fmeasure
#         bleu_scores.append(bleu_score)
#         rouge_scores.append(rouge_score)
#         print(f"Input: {input_text}")
#         print(f"Expected: {expected_response}")
#         print(f"Generated: {generated_response}")
#         print(f"BLEU Score: {bleu_score:.4f}")
#         print(f"ROUGE-L Score: {rouge_score:.4f}\n")
#     except torch.cuda.OutOfMemoryError as e:
#         print(f"CUDA OOM Error during generation: {e}")
#         clear_gpu_memory()
#         continue

# # 10. Calculate average scores
# avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
# avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

# print(f"Average BLEU Score on Test Set: {avg_bleu:.4f}")
# print(f"Average Rouge Score on Test Set: {avg_rouge:.4f}")


# # Clean up
# clear_gpu_memory()