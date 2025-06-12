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
required_columns = ['category', 'subcategory', 'action', 'sentence', 'response']
assert all(col in df.columns for col in required_columns), f"Missing required columns: {required_columns}"

df['Category'] = df['category'].str.lower()
df['Subcategory'] = df['subcategory'].str.lower()
df['stratify_key'] = df['category'] + '_' + df['subcategory'] + '_' + df['action']
combination_counts = df['stratify_key'].value_counts()
valid_combinations = combination_counts[combination_counts >= 2].index
df = df[df['stratify_key'].isin(valid_combinations)]

df['input'] = df.apply(lambda row: f"Category: {row['category']}, Subcategory: {row['subcategory']}, Action: {row['action']}, Sentence: {row['sentence']}", axis=1)
df['output'] = df['response']

try:
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df['stratify_key']
    )
except ValueError:
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df[['input', 'output', 'category', 'subcategory', 'action']])
test_dataset = Dataset.from_pandas(test_df[['input', 'output', 'category', 'subcategory', 'action']])

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
    remove_columns=['input', 'output', 'category', 'subcategory', 'action']
)
test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['input', 'output', 'category', 'subcategory', 'action']
)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_llama",
    per_device_train_batch_size=2,  # Reduced batch size
    per_device_eval_batch_size=4,
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
test_dataset_with_metadata = Dataset.from_pandas(test_df[['input', 'output', 'category', 'subcategory', 'action']])

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
    category = example['category']
    subcategory = example['subcategory']
    action = example['action']
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

