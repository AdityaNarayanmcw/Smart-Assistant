import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback
)
from peft import LoraConfig, get_peft_model
import warnings
from rouge_score import rouge_scorer
from bert_score import score
import numpy as np
import gc
from collections import defaultdict
import re

warnings.filterwarnings("ignore")

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

# Custom callback to log gradients
class GradientLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % args.logging_steps == 0:
            grad_norms = [p.grad.norm().item() if p.grad is not None else None 
                         for n, p in model.named_parameters() if p.requires_grad]
            print(f"Step {state.global_step} - LoRA Gradient Norms: {grad_norms[:10]}...")

# ===============================
# 1. Load and Split Dataset
# ===============================
try:
    df = pd.read_csv("datasetBalanced1.csv")
    print("Sample training data:", df[['sentence', 'response']].head(3))
except FileNotFoundError:
    print("Error: datasetBalanced1.csv not found. Please check the file path.")
    exit(1)

df['stratify_key'] = df['category'] + '_' + df['subcategory'] + '_' + df['action']
try:
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,  # ~473 test rows
        stratify=df['stratify_key'],
        random_state=42
    )
except ValueError:
    print("[INFO] Stratified split failed. Falling back to random split.")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
columns_to_keep = ['sentence', 'response', 'category', 'subcategory', 'action']
train_dataset = Dataset.from_pandas(train_df[columns_to_keep])
test_dataset = Dataset.from_pandas(test_df[columns_to_keep])

# ===============================
# 2. Load Model and Tokenizer
# ===============================
model_name = "./Llama-3.2-3B"  # Update to valid path or Hugging Face model ID
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

if torch.cuda.is_available():
    model = model.to(device="cuda")
else:
    model = model.to("cpu")
    print("[INFO] No GPU available. Using CPU.")

model.gradient_checkpointing_enable()
model.config.pad_token_id = tokenizer.eos_token_id
print_gpu_memory("After Model Loading")

# ===============================
# 3. Apply LoRA
# ===============================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("Trainable parameters after LoRA:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ===============================
# 4. Preprocessing Function
# ===============================
def preprocess_function(examples):
    inputs = [
        f"Instruction: {inp}\nAssistant: {out}"
        for inp, out in zip(examples['sentence'], examples['response'])
    ]
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = model_inputs["input_ids"].clone()
    for i, input_text in enumerate(inputs):
        assistant_idx = input_text.find("Assistant:")
        if assistant_idx != -1:
            input_only = input_text[:assistant_idx]
            input_tokens = tokenizer(input_only, add_special_tokens=False)["input_ids"]
            labels[i, :len(input_tokens) + 2] = -100
    model_inputs["labels"] = labels.tolist()
    return model_inputs

try:
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_keep
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_keep
    )
except Exception as e:
    print(f"Error during dataset preprocessing: {e}")
    exit(1)

# ===============================
# 5. Training Arguments
# ===============================
training_args = TrainingArguments(
    output_dir="./finetuned_llama",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_steps=50,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim="adamw_torch",
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    callbacks=[GradientLoggingCallback()]
)

# Clear memory before training
clear_gpu_memory()

# ===============================
# 6. Train the Model
# ===============================
print("Starting model training...")
try:
    trainer.train()
    print("Training completed.")
except torch.cuda.OutOfMemoryError:
    print("CUDA OOM Error during training. Try reducing LoRA rank to 8.")
    clear_gpu_memory()
    exit(1)
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Check LoRA gradients
lora_grad_norms = [p.grad.norm().item() if p.grad is not None else None 
                   for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]
print("LoRA Gradient Norms:", lora_grad_norms)

try:
    model.save_pretrained("./finetuned_llama/final_model")
    tokenizer.save_pretrained("./finetuned_llama/final_model")
    print("Model and tokenizer saved to ./finetuned_llama/final_model")
except Exception as e:
    print(f"Error saving model: {e}")

# ===============================
# 7. Evaluate Test Performance
# ===============================
def clean_response(text):
    text = text.split(tokenizer.eos_token)[0].split("<|END|>")[0].strip()
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    text = re.sub(r'^(Assistant\.|Instruction:).*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text).strip()
    text = text.replace("checked", "reviewed")
    text = text.split('.')[0].strip() + ('.' if not text.endswith('.') else '')
    return text[:128]

test_dataset_with_metadata = Dataset.from_pandas(test_df[['sentence', 'response', 'category', 'subcategory', 'action']])
rouge_scores = defaultdict(list)
bert_scores = defaultdict(list)
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

try:
    from transformers import RobertaTokenizer, RobertaModel
    bertscore_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    bertscore_model = RobertaModel.from_pretrained('roberta-large')
except Exception as e:
    print(f"Error loading roberta-large for BERTScore: {e}")
    exit(1)

# Evaluate all test samples (~473)
processed_count = 0
batch_size = 4
for i in range(0, len(test_dataset_with_metadata), batch_size):
    indices = range(i, min(i + batch_size, len(test_dataset_with_metadata)))
    batch = test_dataset_with_metadata.select(indices)
    for example in batch:
        input_text = example['sentence']
        expected_response = example['response']
        category = example['category']
        subcategory = example['subcategory']
        action = example['action']
        prompt = f"Instruction: {input_text}\nAssistant: "
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            try:
                generated_response = raw_output.split("Assistant:")[1].strip()
            except IndexError:
                generated_response = raw_output.strip()
            generated_response = clean_response(generated_response)
            
            rouge_score = 0.0
            bert_score_val = 0.0
            if expected_response and generated_response:
                rouge_score = rouge.score(expected_response, generated_response)['rougeL'].fmeasure
                try:
                    P, R, F1 = score([generated_response], [expected_response], lang="en", model_type="roberta-large", verbose=False)
                    bert_score_val = F1.item()
                except Exception as e:
                    print(f"BERTScore error for sentence:\n{input_text}\nError: {e}")

            combination = f"{category}_{subcategory}_{action}"
            rouge_scores[combination].append(rouge_score)
            bert_scores[combination].append(bert_score_val)
            
            print(f"sentence: {input_text}")
            print(f"Expected: {expected_response}")
            print(f"Generated: {generated_response}")
            print(f"ROUGE-L Score: {rouge_score*100:.2f}%")
            print(f"BERTScore: {bert_score_val*100:.2f}%")
            processed_count += 1
            print(f"Processed {processed_count}/{len(test_dataset_with_metadata)} examples\n")
            
            #print_gpu_memory(f"After evaluation sample {processed_count}")
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM Error during generation for sentence: {input_text}")
            #clear_gpu_memory()
            continue
        except Exception as e:
            print(f"Error during evaluation for sentence:\n{input_text}\nError: {e}")
            continue

# ===============================
# 8. Calculate and Save Metrics
# ===============================
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

rouge_values = [entry['ROUGE-L (%)'] for entry in results]
bert_values = [entry['BERTScore (%)'] for entry in results]
avg_rouge = np.mean(rouge_values) if rouge_values else 0.0
avg_bert = np.mean(bert_values) if bert_values else 0.0

print(f"Average ROUGE-L Score: {avg_rouge:.2f}%")
print(f"Average BERTScore: {avg_bert:.2f}%")

if results:
    print(f"Number of results: {len(results)}")
    print(f"Sample result: {results[:1] if results else 'None'}")
    results_df = pd.DataFrame(results)
    print("\nMetrics by Category, Subcategory, and Action:")
    print(results_df.to_string(index=False))

    try:
        os.makedirs(os.path.dirname('metrics_by_combination.csv') or '.', exist_ok=True)
        pd.DataFrame({'test': [1]}).to_csv('test.csv')
        print("Test CSV written successfully")
        results_df.to_csv('metrics_by_combination.csv', index=False)
        print("\nResults saved to 'metrics_by_combination.csv'")
    except PermissionError:
        print("Error: Permission denied when saving 'metrics_by_combination.csv'. Check directory permissions.")
    except OSError as e:
        print(f"Error: Failed to save 'metrics_by_combination.csv' due to OS issue: {e}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
else:
    print("\nNo results to save. Evaluation loop produced no metrics.")

try:
    clear_gpu_memory()
except Exception as e:
    print(f"Error clearing GPU memory: {e}")
