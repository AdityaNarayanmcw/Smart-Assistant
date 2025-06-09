import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from bert_score import score
import gc
from collections import defaultdict
import numpy as np
import re

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_memory("After Clearing GPU Memory")

clear_gpu_memory()

# 1. Prepare the dataset
data = "./datasetBalanced1.csv"
df = pd.read_csv(data)
required_columns = ['Category', 'Subcategory', 'Action', 'Sentence', 'Response']
assert all(col in df.columns for col in required_columns), f"Missing required columns: {required_columns}"

df['Category'] = df['Category'].str.lower()
df['Subcategory'] = df['Subcategory'].str.lower()
df['stratify_key'] = df['Category'] + '_' + df['Subcategory'] + '_' + df['Action']
combination_counts = df['stratify_key'].value_counts()
valid_combinations = combination_counts[combination_counts >= 2].index
df = df[df['stratify_key'].isin(valid_combinations)]

df['input'] = df.apply(lambda row: f"Category: {row['Category']}, Subcategory: {row['Subcategory']}, Action: {row['Action']}, Sentence: {row['Sentence']}", axis=1)
df['output'] = df['Response']

try:
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df['stratify_key']
    )
except ValueError:
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df[['input', 'output', 'Category', 'Subcategory', 'Action']])
test_dataset = Dataset.from_pandas(test_df[['input', 'output', 'Category', 'Subcategory', 'Action']])

# 2. Load model and tokenizer
model_name = "./Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_state_dict=True,
)
model.gradient_checkpointing_enable()
print_gpu_memory("After Loading Base Model")

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
    inputs = [
        f"Instruction: {inp}\nRespond concisely with the exact action result, without timestamps or extra details.\nAssistant: {out}"
        for inp, out in zip(examples['input'], examples['output'])
    ]
    model_inputs = tokenizer(
        inputs,
        max_length=128,  # Reduced max_length for memory efficiency
        padding="max_length",
        truncation=True,
    )

    input_ids = model_inputs["input_ids"]
    labels = []

    for input_id in input_ids:
        label = [-100 if token == tokenizer.pad_token_id else token for token in input_id]
        labels.append(label)

    model_inputs["labels"] = labels
    return model_inputs

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['input', 'output', 'Category', 'Subcategory', 'Action']
)
test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['input', 'output', 'Category', 'Subcategory', 'Action']
)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_llama",
    per_device_train_batch_size=2,  # Reduced batch size
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=7,  # Increased epochs for better training
    learning_rate=1e-4,  # Slightly lower learning rate for stability
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# 7. Train
trainer.train()

# 8. Save the fine-tuned model
model.save_pretrained("./finetuned_llama2", safe_serialization=True)
tokenizer.save_pretrained("./finetuned_llama2")

# 9. Evaluate test performance per category, subcategory, and action
def clean_response(text):
    text = text.split(tokenizer.eos_token)[0].split("<|END|>")[0].strip()
    text = re.sub(r'\s+([.,;:])', r'\1', text)  # Fix punctuation spacing
    text = re.sub(r'^(Assistant\.|Instruction:).*', '', text, flags=re.IGNORECASE).strip()
    # Remove timestamps (e.g., "2023-08-09 09:45:07")
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text).strip()
    # Normalize synonyms (e.g., "checked" to "reviewed")
    text = text.replace("checked", "reviewed")
    # Keep only the first sentence
    text = text.split('.')[0].strip() + ('.' if not text.endswith('.') else '')
    return text[:128]

# Load test dataset with original columns
test_dataset_with_metadata = Dataset.from_pandas(test_df[['input', 'output', 'Category', 'Subcategory', 'Action']])

rouge_scores = defaultdict(list)
bert_scores = defaultdict(list)
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Explicitly load roberta-large for BERTScore
from transformers import RobertaTokenizer, RobertaModel
bertscore_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
bertscore_model = RobertaModel.from_pretrained('roberta-large')

for example in test_dataset_with_metadata:
    input_text = example['input']
    expected_response = example['output']
    category = example['Category']
    subcategory = example['Subcategory']
    action = example['Action']
    prompt = f"Instruction: {input_text}\nRespond concisely and only with the action result. , without timestamps or extra details.\nAssistant: "
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # Reduced to enforce brevity
                num_beams=5,
                do_sample=False,  # Disable sampling for deterministic output
                temperature=0.3,  # Lower temperature for more focused output
                top_p=0.9,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
        generated_response = clean_response(generated_response)
        print(f"Input: {input_text}")
        print(f"Expected: {expected_response}")
        print(f"Generated: {generated_response}")
        
        # Calculate ROUGE-L
        if expected_response is None or generated_response is None:
            rouge_score = 0.0
            bert_score_val = 0.0
        else:
            rouge_score = rouge.score(expected_response, generated_response)['rougeL'].fmeasure

        # Safe BERTScore computation
        try:
            P, R, F1 = score([generated_response], [expected_response], lang="en", model_type="roberta-large", verbose=False)
            bert_score_val = F1.item()
        except Exception as e:
            print(f"BERTScore error for input:\n{input_text}\nError: {e}")
            bert_score_val = 0.0

        combination = f"{category}_{subcategory}_{action}"
        rouge_scores[combination].append(rouge_score)
        bert_scores[combination].append(bert_score_val)
        print(f"ROUGE-L Score: {rouge_score*100:.2f}%")
        print(f"BERTScore: {bert_score_val*100:.2f}%\n")
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error during generation: {e}")
        clear_gpu_memory()
        continue

# 10. Calculate and display metrics by combination
results = []
for combo in sorted(rouge_scores.keys()):
    rouge_avg = np.mean(rouge_scores[combo]) * 100
    bert_avg = np.mean(bert_scores[combo]) * 100
    count = len(rouge_scores[combo])
    category, subcategory, action = combo.split('_')
    results.append({
        'Category': category,
        'Subcategory': subcategory,
        'Action': action,
        'Count': count,
        'ROUGE-L (%)': round(rouge_avg, 2),
        'BERTScore (%)': round(bert_avg, 2)
    })
# Extract ROUGE-L and BERTScore values
rouge_values = [entry['ROUGE-L (%)'] for entry in results]
bert_values = [entry['BERTScore (%)'] for entry in results]

# Calculate averages
avg_rouge = np.mean(rouge_values) if rouge_values else 0.0
avg_bert = np.mean(bert_values) if bert_values else 0.0

# Print results
print(f"Average ROUGE-L Score: {avg_rouge:.2f}%")
print(f"Average BERTScore: {avg_bert:.2f}%")

results_df = pd.DataFrame(results)
print("\nMetrics by Category, Subcategory, and Action:")
print(results_df.to_string(index=False))
results_df.to_csv('metrics_by_combination.csv', index=False)
print("\nResults saved to 'metrics_by_combination.csv'")

clear_gpu_memory()


#----------------------------------------------------------------
# import os
# import pandas as pd
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# from peft import LoraConfig, get_peft_model
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from rouge_score import rouge_scorer
# from bert_score import score
# import gc
# from collections import defaultdict
# import numpy as np
# import re
# from accelerate import Accelerator

# # Set environment variable to reduce memory fragmentation
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# print(f"Number of CUDA devices: {torch.cuda.device_count()}")

# # Function to check GPU memory usage
# def print_gpu_memory(step_name):
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#         allocated = torch.cuda.memory_allocated(0) / 1e9
#         reserved = torch.cuda.memory_reserved(0) / 1e9
#         total = torch.cuda.get_device_properties(0).total_memory / 1e9
#         print(f"\nMemory Usage at {step_name}:")
#         print(f"  Allocated: {allocated:.2f} GiB")
#         print(f"  Reserved: {reserved:.2f} GiB")
#         print(f"  Free: {total - allocated:.2f} GiB")
#         print(f"  Total: {total:.2f} GiB")

# # Clear GPU memory
# def clear_gpu_memory():
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         gc.collect()
#         print_gpu_memory("After Clearing GPU Memory")

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

# # Verify required columns
# required_columns = ['Category', 'Subcategory', 'Action', 'Sentence', 'Response']
# if not all(col in df.columns for col in required_columns):
#     print(f"Error: Dataset must contain {required_columns}. Found: {df.columns.tolist()}")
#     raise ValueError("Missing required columns.")

# # Normalize category names
# df['Category'] = df['Category'].str.lower()
# df['Subcategory'] = df['Subcategory'].str.lower()

# # Filter out combinations with fewer than 2 occurrences
# df['stratify_key'] = df['Category'] + '_' + df['Subcategory'] + '_' + df['Action']
# combination_counts = df['stratify_key'].value_counts()
# valid_combinations = combination_counts[combination_counts >= 2].index
# df = df[df['stratify_key'].isin(valid_combinations)]
# print(f"\nAfter filtering, dataset size: {len(df)}")
# print("Remaining combination counts:")
# print(df['stratify_key'].value_counts())

# # Create input prompt
# df['input'] = df.apply(lambda row: f"Category: {row['Category']}, Subcategory: {row['Subcategory']}, Action: {row['Action']}, Sentence: {row['Sentence']}", axis=1)
# df['output'] = df['Response']

# # Split into train and test sets
# try:
#     train_df, test_df = train_test_split(
#         df,
#         test_size=0.1,
#         random_state=42,
#         stratify=df['stratify_key']
#     )
# except ValueError as e:
#     print(f"Stratification failed: {e}")
#     print("Falling back to non-stratified split.")
#     train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# train_dataset = Dataset.from_pandas(train_df[['input', 'output', 'Category', 'Subcategory', 'Action']])
# test_dataset = Dataset.from_pandas(test_df[['input', 'output', 'Category', 'Subcategory', 'Action']])

# # 2. Load model and tokenizer
# model_name = "./Llama-3.2-3B"
# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#         offload_state_dict=True,
#     )
#     model.gradient_checkpointing_enable()
#     print_gpu_memory("After Loading Base Model")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     raise

# # Set padding token
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.eos_token_id

# # 3. Apply LoRA
# lora_config = LoraConfig(
#     r=16,  # Increased rank for better capacity
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj", "k_proj"],
#     lora_dropout=0.2,  # Increased dropout
#     bias="none",
#     task_type="CAUSAL_LM"
# )
# model = get_peft_model(model, lora_config)
# print_gpu_memory("After Applying LoRA")

# # 4. Preprocess dataset
# def preprocess_function(examples):
#     inputs = [f"Instruction: {inp}\nAssistant: {out} <|END|>" for inp, out in zip(examples['input'], examples['output'])]
#     tokenized = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
#     tokenized['labels'] = tokenized['input_ids'].copy()
#     return tokenized

# train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['input', 'output', 'Category', 'Subcategory', 'Action'])
# test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=['input', 'output', 'Category', 'Subcategory', 'Action'])

# # 5. Initialize Accelerator
# accelerator = Accelerator()
# print(accelerator.state)
# model, train_dataset, test_dataset = accelerator.prepare(model, train_dataset, test_dataset)

# # 6. Training arguments
# training_args = TrainingArguments(
#     output_dir="./finetuned_llama",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=2,
#     num_train_epochs=10,  # Increased epochs
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=10,
#     save_strategy="epoch",
#     eval_strategy="epoch",
#     save_total_limit=1,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     lr_scheduler_type="cosine",
#     warmup_steps=100,  # Increased warmup
#     gradient_checkpointing=True,
#     optim="adamw_8bit",
# )

# # 7. Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )

# # 8. Fine-tune the model
# try:
#     trainer.train()
#     print_gpu_memory("After Training")
# except torch.cuda.OutOfMemoryError as e:
#     print(f"CUDA OOM Error during training: {e}")
#     clear_gpu_memory()
#     raise

# # 9. Save the fine-tuned model
# model.save_pretrained("./finetuned_llama2")
# tokenizer.save_pretrained("./finetuned_llama2")

# # 10. Evaluate test performance per category, subcategory, and action
# model.eval()
# rouge_scores = defaultdict(list)
# bert_scores = defaultdict(list)
# rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# def clean_response(text):
#     text = text.split("<|END|>")[0].strip()
#     text = re.sub(r'(?<!\d)\s+', ' ', text).strip('.,:; ')
#     text = re.sub(r'^(Assistant\.|Instruction:).*', '', text, flags=re.IGNORECASE).strip()
#     return text

# def post_process_response(generated, category, subcategory, action):
#     if category == 'music' and action == 'volume_sync':
#         return f"Volume synced for {subcategory}."
#     elif category == 'camera' and 'at' in generated.lower():
#         time_match = re.search(r'\d{1,2}:\d{2}\s?(AM|PM|am|pm)', generated)
#         if time_match:
#             return f"Camera will start at {time_match.group(0)}."
#     elif category == 'lights' and action == 'no_action':
#         return f"{subcategory.capitalize()} lighting status shown."
#     elif category == 'music' and action == 'play' and subcategory == 'genre':
#         genre_match = re.search(r'\b(jazz|classical|pop|rock|reggae|hip-hop|blues|country|indie|electronic)\b', generated, re.IGNORECASE)
#         if genre_match:
#             return f"{genre_match.group(0).capitalize()} music playing."
#     return generated

# # Load test dataset with original columns
# test_dataset_with_metadata = Dataset.from_pandas(test_df[['input', 'output', 'Category', 'Subcategory', 'Action']])

# for example in test_dataset_with_metadata:
#     input_text = example['input']
#     expected_response = example['output']
#     category = example['Category']
#     subcategory = example['Subcategory']
#     action = example['Action']
#     prompt = f"Instruction: {input_text}\nAssistant: "
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=50,
#             num_beams=3,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             no_repeat_ngram_size=3,
#         )
#         generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
#         print(generated_response)
#         generated_response = clean_response(generated_response)
#         generated_response = post_process_response(generated_response, category, subcategory, action)
        
#         # Calculate ROUGE-L
#         if expected_response is None or generated_response is None:
#             print(f"Warning: Missing response. Expected: {expected_response}, Generated: {generated_response}")
#         else:
#             rouge_score = rouge_scorer.score(expected_response, generated_response)['rougeL'].fmeasure

#         # rouge_score = rouge_scorer.score(expected_response, generated_response)['rougeL'].fmeasure
#         # Calculate BERTScore
#         P, R, F1 = score([generated_response], [expected_response], lang="en", verbose=False)
#         bert_score = F1.item()
        
#         combination = f"{category}_{subcategory}_{action}"
#         rouge_scores[combination].append(rouge_score)
#         bert_scores[combination].append(bert_score)
        
#         print(f"Input: {input_text}")
#         print(f"Expected: {expected_response}")
#         print(f"Generated: {generated_response}")
#         print(f"ROUGE-L Score: {rouge_score*100:.2f}%")
#         print(f"BERTScore: {bert_score*100:.2f}%\n")
#     except torch.cuda.OutOfMemoryError as e:
#         print(f"CUDA OOM Error during generation: {e}")
#         clear_gpu_memory()
#         continue

# # 11. Calculate and display metrics by combination
# results = []
# for combo in sorted(rouge_scores.keys()):
#     rouge_avg = np.mean(rouge_scores[combo]) * 100  # Convert to percentage
#     bert_avg = np.mean(bert_scores[combo]) * 100    # Convert to percentage
#     count = len(rouge_scores[combo])
#     category, subcategory, action = combo.split('_')
#     results.append({
#         'Category': category,
#         'Subcategory': subcategory,
#         'Action': action,
#         'Count': count,
#         'ROUGE-L (%)': round(rouge_avg, 2),
#         'BERTScore (%)': round(bert_avg, 2)
#     })

# results_df = pd.DataFrame(results)
# print("\nMetrics by Category, Subcategory, and Action:")
# print(results_df.to_string(index=False))
# results_df.to_csv('metrics_by_combination.csv', index=False)
# print("\nResults saved to 'metrics_by_combination.csv'")

# # Clean up
# clear_gpu_memory()

# import os
# import pandas as pd
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# from peft import LoraConfig, get_peft_model
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer
# import gc
# from collections import defaultdict
# import numpy as np
# import re
# from accelerate import Accelerator

# # Set environment variable to reduce memory fragmentation
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# print(torch.cuda.device_count())  # Should print 2

# # Function to check GPU memory usage
# def print_gpu_memory(step_name):
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#         allocated = torch.cuda.memory_allocated(0) / 1e9
#         reserved = torch.cuda.memory_reserved(0) / 1e9
#         total = torch.cuda.get_device_properties(0).total_memory / 1e9
#         print(f"\nMemory Usage at {step_name}:")
#         print(f"  Allocated: {allocated:.2f} GiB")
#         print(f"  Reserved: {reserved:.2f} GiB")
#         print(f"  Free: {total - allocated:.2f} GiB")
#         print(f"  Total: {total:.2f} GiB")

# # Clear GPU memory
# def clear_gpu_memory():
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         gc.collect()
#         print_gpu_memory("After Clearing GPU Memory")

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
# # print("\nSubcategory counts:")
# # print(df['Subcategory'].value_counts())
# # print("\nAction counts:")
# # print(df['Action'].value_counts())

# # Verify required columns
# required_columns = ['Category', 'Subcategory', 'Action', 'Sentence', 'Response']
# if not all(col in df.columns for col in required_columns):
#     print(f"Error: Dataset must contain {required_columns}. Found: {df.columns.tolist()}")
#     raise ValueError("Missing required columns.")

# # Normalize category names (e.g., 'Camera' and 'camera' to 'camera')
# df['Category'] = df['Category'].str.lower()
# df['Subcategory'] = df['Subcategory'].str.lower()

# # Filter out combinations with fewer than 2 occurrences
# df['stratify_key'] = df['Category'] + '_' + df['Subcategory'] + '_' + df['Action']
# combination_counts = df['stratify_key'].value_counts()
# valid_combinations = combination_counts[combination_counts >= 2].index
# df = df[df['stratify_key'].isin(valid_combinations)]
# print(f"\nAfter filtering, dataset size: {len(df)}")
# print("Remaining combination counts:")
# print(df['stratify_key'].value_counts())

# # Create input prompt combining Category, Subcategory, Action, and Sentence
# df['input'] = df.apply(lambda row: f"Category: {row['Category']}, Subcategory: {row['Subcategory']}, Action: {row['Action']}, Sentence: {row['Sentence']}", axis=1)
# df['output'] = df['Response']

# # Split into train and test sets
# try:
#     train_df, test_df = train_test_split(
#         df,
#         test_size=0.1,
#         random_state=42,
#         stratify=df['stratify_key']
#     )
# except ValueError as e:
#     print(f"Stratification failed: {e}")
#     print("Falling back to non-stratified split.")
#     train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# train_dataset = Dataset.from_pandas(train_df[['input', 'output', 'Category', 'Subcategory', 'Action']])
# test_dataset = Dataset.from_pandas(test_df[['input', 'output', 'Category', 'Subcategory', 'Action']])

# # 2. Load model and tokenizer
# model_name = "./Llama-3.2-3B"
# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#         offload_state_dict=True,
#     )
#     model.gradient_checkpointing_enable()
#     print_gpu_memory("After Loading Base Model")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     raise

# # Set padding token
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.eos_token_id

# # 3. Apply LoRA
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=["q_proj", "v_proj", "k_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
# model = get_peft_model(model, lora_config)
# print_gpu_memory("After Applying LoRA")

# # 4. Preprocess dataset
# def preprocess_function(examples):
#     inputs = [f"Instruction: {inp}\nAssistant: {out} <|END|>" for inp, out in zip(examples['input'], examples['output'])]
#     tokenized = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
#     tokenized['labels'] = tokenized['input_ids'].copy()
#     return tokenized

# train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['input', 'output', 'Category', 'Subcategory', 'Action'])
# test_dataset = test_dataset.map(preprocess_function, batched=True)

# accelerator=Accelerator()
# print(accelerator.state)
# model,train_dataset,test_dataset=accelerator.prepare(model,train_dataset,test_dataset)
# # 5. Training arguments
# training_args = TrainingArguments(
#     output_dir="./finetuned_llama",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=2,
#     num_train_epochs=5,
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=10,
#     save_strategy="epoch",  # Changed to "epoch"
#     eval_strategy="epoch",
#     save_total_limit=1,
#     load_best_model_at_end=True,
#     lr_scheduler_type="cosine",
#     warmup_steps=50,
#     gradient_checkpointing=True,
#     optim="adamw_8bit",
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
# model.save_pretrained("./finetuned_llama2")
# tokenizer.save_pretrained("./finetuned_llama2")

# # 9. Evaluate test accuracy per category, subcategory, and action
# model.eval()
# bleu_scores = defaultdict(list)
# rouge_scores = defaultdict(list)
# exact_matches = defaultdict(list)
# rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# smoother = SmoothingFunction().method1

# def clean_response(text):
#     """Clean generated response to remove extraneous text."""
#     # Remove anything after <|END|> or other metadata
#     text = text.split("<|END|>")[0].strip()
#     # Remove patterns like 'Month, Day, Year', 'Assistant.', etc.
#     text = re.sub(r'(Month|Day|Year|Hour|Minute|Second|Weekday|Temperature|Humidity|.*\bAssistant\b.*$)', '', text, flags=re.IGNORECASE)
#     # Remove extra punctuation, spaces, and trailing text
#     text = re.sub(r'\s+', ' ', text).strip('.,:; ')
#     return text

# for example in test_dataset:
#     input_text = example['input']
#     expected_response = example['output']
#     category = example['Category']
#     subcategory = example['Subcategory']
#     action = example['Action']
#     prompt = f"Instruction: {input_text}\nAssistant: "
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=30,  # Reduced to enforce concise responses
#             num_beams=5,
#             no_repeat_ngram_size=3,
#             do_sample=False,
#         )
#         generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
#         generated_response = clean_response(generated_response)
#         bleu_score = sentence_bleu([expected_response.split()], generated_response.split(), smoothing_function=smoother)
#         rouge_score = rouge_scorer.score(expected_response, generated_response)['rougeL'].fmeasure
#         exact_match = 1 if generated_response.lower() == expected_response.lower() else 0
#         bleu_scores[('Category', category)].append(bleu_score)
#         bleu_scores[('Subcategory', subcategory)].append(bleu_score)
#         bleu_scores[('Action', action)].append(bleu_score)
#         rouge_scores[('Category', category)].append(rouge_score)
#         rouge_scores[('Subcategory', subcategory)].append(rouge_score)
#         rouge_scores[('Action', action)].append(rouge_score)
#         exact_matches[('Category', category)].append(exact_match)
#         exact_matches[('Subcategory', subcategory)].append(exact_match)
#         exact_matches[('Action', action)].append(exact_match)
#         print(f"Input: {input_text}")
#         print(f"Expected: {expected_response}")
#         print(f"Generated: {generated_response}")
#         print(f"BLEU Score: {bleu_score:.4f}")
#         print(f"ROUGE-L Score: {rouge_score:.4f}")
#         print(f"Exact Match: {exact_match}\n")
#     except torch.cuda.OutOfMemoryError as e:
#         print(f"CUDA OOM Error during generation: {e}")
#         clear_gpu_memory()
#         continue

# # 10. Calculate and display average scores per category, subcategory, and action
# print("\nEvaluation Metrics by Category, Subcategory, and Action:")
# for key, scores in sorted(bleu_scores.items()):
#     group_type, group_name = key
#     avg_bleu = np.mean(scores)
#     avg_rouge = np.mean(rouge_scores[key])
#     avg_exact = np.mean(exact_matches[key])
#     count = len(scores)
#     print(f"\n{group_type}: {group_name} (Count: {count})")
#     print(f"  Average BLEU Score: {avg_bleu:.4f}")
#     print(f"  Average ROUGE-L Score: {avg_rouge:.4f}")
#     print(f"  Exact Match Accuracy: {avg_exact:.4f}")

# # 11. Overall metrics
# avg_bleu = np.mean([score for scores in bleu_scores.values() for score in scores])
# avg_rouge = np.mean([score for scores in rouge_scores.values() for score in scores])
# avg_exact = np.mean([match for matches in exact_matches.values() for match in matches])
# print(f"\nOverall Average BLEU Score: {avg_bleu:.4f}")
# print(f"Overall Average ROUGE-L Score: {avg_rouge:.4f}")
# print(f"Overall Exact Match Accuracy: {avg_exact:.4f}")

# # Clean up
# clear_gpu_memory()

