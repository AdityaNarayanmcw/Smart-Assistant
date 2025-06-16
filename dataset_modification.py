import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from collections import defaultdict
import re
import gc
from tqdm import tqdm
import warnings
try:
    import spacy
    from scipy.spatial.distance import cosine
except ImportError:
    spacy = None
    cosine = None
import time
import uuid

warnings.filterwarnings("ignore")

# Define cache directory
CACHE_DIR = "./dataset_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Memory management function
def manage_gpu_memory(step_name, clear=False):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if clear:
            torch.cuda.empty_cache()
            gc.collect()
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nMemory at {step_name}: Allocated={allocated:.2f}GiB, Reserved={reserved:.2f}GiB, Free={total-allocated:.2f}GiB")

# Custom callback to log gradients
class GradientLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % args.logging_steps == 0:
            grad_norms = [p.grad.norm().item() if p.grad is not None else None 
                         for n, p in model.named_parameters() if p.requires_grad]
            print(f"Step {state.global_step} - LoRA Gradient Norms: {grad_norms[:10]}...")

# Sentence normalization and cosine similarity
intent_map = defaultdict(lambda: "command")
query_actions = ["statusquery", "check"]
devices = ["camera", "lights", "music", "shutters"]
locations = ["attic", "backyard", "basement", "bathroom", "cellar", "kitchen", "library", 
             "living room", "none", "outside", "restroom", "toilet", "dining room"]

# Load spaCy with error handling
nlp = None
if spacy:
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model 'en_core_web_sm'")
    except Exception as e:
        print(f"Error loading spaCy model 'en_core_web_sm': {e}")
        print("Falling back to basic tokenization for normalization")
else:
    print("spaCy not installed. Falling back to basic tokenization for normalization")

def normalize_sentence(sentence, category=None, subcategory=None, action=None):
    """
    Normalize a sentence into a structured meaning representation.
    Args:
        sentence (str): Input sentence.
        category, subcategory, action (str): Metadata from dataset.
    Returns:
        dict: Structured representation with intent, action, device, location.
    """
    if nlp:
        doc = nlp(sentence.lower())
    else:
        # Fallback: simple tokenization
        doc = sentence.lower().split()
    
    intent = "unknown"
    device = category.lower() if category else "none"
    location = subcategory.lower() if subcategory else "none"
    action_slot = action.lower() if action else "none"
    
    if action_slot in intent_map:
        intent = intent_map[action_slot]
    else:
        # Fallback intent classification
        command_verbs = ["turn", "activate", "play", "stop", "adjust"]
        query_verbs = ["check", "query", "status"]
        for token in doc if nlp else doc:
            token_text = token.text if nlp else token
            if token_text in command_verbs:
                intent = "command"
            elif token_text in query_verbs:
                intent = "query"
    
    if not category:
        for token in doc if nlp else doc:
            token_text = token.text if nlp else token
            if token_text in devices:
                device = token_text
                break
    if not subcategory:
        for token in doc if nlp else doc:
            token_text = token.text if nlp else token
            if token_text in locations:
                location = token_text
                break
    
    return {
        "intent": intent,
        "action": action_slot,
        "device": device,
        "location": location
    }

def representation_to_vector(rep):
    """
    Convert structured representation to a vector using spaCy embeddings or one-hot encoding.
    Args:
        rep (dict): Structured representation.
    Returns:
        np.array: Vector representation.
    """
    if nlp and spacy:
        intent_vec = nlp(rep["intent"]).vector
        action_vec = nlp(rep["action"]).vector
        device_vec = nlp(rep["device"]).vector
        location_vec = nlp(rep["location"]).vector
        return np.concatenate([intent_vec, action_vec, device_vec, location_vec])
    else:
        # Fallback: one-hot encoding for intent, action, device, location
        all_intents = ["command", "query", "unknown"]
        all_actions = list(set(df['action'].str.lower())) if 'df' in globals() else devices + locations + query_actions
        all_devices = devices
        all_locations = locations
        
        intent_idx = all_intents.index(rep["intent"]) if rep["intent"] in all_intents else len(all_intents)
        action_idx = all_actions.index(rep["action"]) if rep["action"] in all_actions else len(all_actions)
        device_idx = all_devices.index(rep["device"]) if rep["device"] in all_devices else len(all_devices)
        location_idx = all_locations.index(rep["location"]) if rep["location"] in all_locations else len(all_locations)
        
        vec = np.zeros(len(all_intents) + len(all_actions) + len(all_devices) + len(all_locations))
        vec[intent_idx] = 1
        vec[len(all_intents) + action_idx] = 1
        vec[len(all_intents) + len(all_actions) + device_idx] = 1
        vec[len(all_intents) + len(all_actions) + len(all_devices) + location_idx] = 1
        return vec

def compute_cosine_similarity(rep1, rep2):
    """
    Compute cosine similarity between two structured representations.
    Args:
        rep1, rep2 (dict): Structured representations.
    Returns:
        float: Cosine similarity score (0 to 1).
    """
    if spacy and cosine and nlp:
        vec1 = representation_to_vector(rep1)
        vec2 = representation_to_vector(rep2)
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        return 1 - cosine(vec1, vec2)
    else:
        # Fallback: cosine similarity on one-hot vectors
        vec1 = representation_to_vector(rep1)
        vec2 = representation_to_vector(rep2)
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0.0

# Load and split dataset
try:
    df = pd.read_csv("datasetBalanced1.csv")
    print("Sample training data:", df[['sentence', 'response']].head(3))
except FileNotFoundError:
    print("Error: datasetBalanced1.csv not found. Please check the file path.")
    exit(1)

# Dynamic intent mapping
for action in df['action'].unique():
    if any(q in action.lower() for q in query_actions):
        intent_map[action] = "query"

df['stratify_key'] = df['category'] + '_' + df['subcategory'] + '_' + df['action']
try:
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        stratify=df['stratify_key'],
        random_state=42
    )
except ValueError:
    print("[INFO] Stratified split failed. Falling back to random split.")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
columns_to_keep = ['sentence', 'response', 'category', 'subcategory', 'action']

# Create datasets with caching
try:
    train_dataset = Dataset.from_pandas(train_df[columns_to_keep]).map(
        lambda x: x,
        cache_file_name=os.path.join(CACHE_DIR, "train_cache.arrow")
    )
    test_dataset = Dataset.from_pandas(test_df[columns_to_keep]).map(
        lambda x: x,
        cache_file_name=os.path.join(CACHE_DIR, "test_cache.arrow")
    )
except Exception as e:
    print(f"Error caching dataset: {e}")
    print("Falling back to in-memory processing")
    train_dataset = Dataset.from_pandas(train_df[columns_to_keep])
    test_dataset = Dataset.from_pandas(test_df[columns_to_keep])

# Load model and tokenizer
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
manage_gpu_memory("After Model Loading")

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True
)
model = get_peft_model(model, lora_config)
print("Trainable parameters after LoRA:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Preprocessing function
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
        remove_columns=columns_to_keep,
        cache_file_name=os.path.join(CACHE_DIR, "train_cache_preprocessed.arrow")
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_keep,
        cache_file_name=os.path.join(CACHE_DIR, "test_cache_preprocessed.arrow")
    )
except Exception as e:
    print(f"Error during dataset preprocessing: {e}")
    print("Falling back to in-memory preprocessing")
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

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_llama",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=5,
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

# Train the model
manage_gpu_memory("Before Training", clear=True)
print("Starting model training...")
try:
    trainer.train()
    print("Training completed.")
except torch.cuda.OutOfMemoryError:
    print("CUDA OOM Error during training. Try reducing batch size or LoRA rank.")
    manage_gpu_memory("After OOM", clear=True)
    exit(1)
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Save model and tokenizer
try:
    model.save_pretrained("./finetuned_llama/final_model")
    tokenizer.save_pretrained("./finetuned_llama/final_model")
    print("Model and tokenizer saved to ./finetuned_llama/final_model")
except Exception as e:
    print(f"Error saving model: {e}")

# Clean response function
def clean_response(text):
    text = text.split(tokenizer.eos_token)[0].split("<|END|>")[0].strip()
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    text = re.sub(r'^(Assistant\.|Instruction:).*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text).strip()
    text = text.replace("checked", "reviewed")
    text = text.split('.')[0].strip() + ('.' if not text.endswith('.') else '')
    return text[:128]

# Evaluation function
def evaluate_sample(input_text, expected_response, category, subcategory, action, model, tokenizer):
    try:
        prompt = f"Instruction: {input_text}\nAssistant: "
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
                pad_token_id=tokenizer.eos_token_id
            )
        generated_response = clean_response(tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[1].strip())
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_score = rouge.score(expected_response, generated_response)['rougeL'].fmeasure if expected_response and generated_response else 0.0
        expected_norm = normalize_sentence(expected_response, category, subcategory, action)
        generated_norm = normalize_sentence(generated_response, category, subcategory, action)
        cosine_sim = compute_cosine_similarity(expected_norm, generated_norm)
        return rouge_score, cosine_sim, generated_response
    except Exception as e:
        print(f"Error evaluating sentence: {input_text}\nError: {e}")
        return 0.0, 0.0, ""

# Evaluate test performance
test_dataset_with_metadata = Dataset.from_pandas(test_df[['sentence', 'response', 'category', 'subcategory', 'action']])
rouge_scores = defaultdict(list)
cosine_scores = defaultdict(list)
test_dataloader = DataLoader(test_dataset_with_metadata, batch_size=4, shuffle=False)

for batch in tqdm(test_dataloader, desc="Evaluating"):
    input_texts = batch['sentence']
    expected_responses = batch['response']
    categories = batch['category']
    subcategories = batch['subcategory']
    actions = batch['action']
    
    try:
        for i, (input_text, expected, category, subcategory, action) in enumerate(zip(
            input_texts, expected_responses, categories, subcategories, actions
        )):
            rouge_score, cosine_sim, generated_response = evaluate_sample(
                input_text, expected, category, subcategory, action, model, tokenizer
            )
            combination = f"{category}_{subcategory}_{action}"
            rouge_scores[combination].append(rouge_score)
            cosine_scores[combination].append(cosine_sim)
            
            print(f"Sentence: {input_text}")
            print(f"Expected: {expected}")
            print(f"Generated: {generated_response}")
            print(f"ROUGE-L: {rouge_score*100:.2f}%")
            print(f"Cosine Similarity: {cosine_sim*100:.2f}%")
            print(f"Processed {sum(len(v) for v in rouge_scores.values())}/{len(test_dataset_with_metadata)} examples\n")
        
        manage_gpu_memory("After batch evaluation", clear=True)
    except Exception as e:
        print(f"Error processing batch: {e}")
        continue

# Calculate and save metrics
results = []
for combo in sorted(rouge_scores.keys()):
    rouge_avg = np.mean(rouge_scores[combo]) * 100
    cosine_avg = np.mean(cosine_scores[combo]) * 100
    count = len(rouge_scores[combo])
    category, subcategory, action = combo.split('_')
    results.append({
        'Category': category,
        'Subcategory': subcategory,
        'Action': action,
        'Count': count,
        'ROUGE-L (%)': round(rouge_avg, 2),
        'Cosine Similarity (%)': round(cosine_avg, 2)
    })

rouge_values = [entry['ROUGE-L (%)'] for entry in results]
cosine_values = [entry['Cosine Similarity (%)'] for entry in results]
avg_rouge = np.mean(rouge_values) if rouge_values else 0.0
avg_cosine = np.mean(cosine_values) if cosine_values else 0.0

print(f"Average ROUGE-L Score: {avg_rouge:.2f}%")
print(f"Average Cosine Similarity: {avg_cosine:.2f}%")

# Save metrics
def save_metrics(results, filename):
    for attempt in range(3):
        try:
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            pd.DataFrame(results).to_csv(filename, index=False)
            print(f"Results saved to {filename}")
            return
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    print(f"Failed to save {filename} after 3 attempts.")

if results:
    print(f"Number of results: {len(results)}")
    print(f"Sample result: {results[:1] if results else 'None'}")
    print("\nMetrics by Category, Subcategory, and Action:")
    print(pd.DataFrame(results).to_string(index=False))
    save_metrics(results, "metrics_by_combination.csv")
else:
    print("\nNo results to save. Evaluation loop produced no metrics.")

manage_gpu_memory("After Evaluation", clear=True)

#------------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from sentence_transformers import SentenceTransformer, util
# from rouge_score import rouge_scorer
# from bert_score import score
# from Levenshtein import distance as levenshtein_distance
# from collections import defaultdict
# import re
# import warnings
# import gc

# warnings.filterwarnings("ignore")

# # Helper function to print GPU memory usage
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

# # Helper function to clear GPU memory
# def clear_gpu_memory():
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         gc.collect()
#         print_gpu_memory("After Clearing GPU Memory")

# # Enhanced function to normalize sentences into structured representations
# def normalize_sentence(sentence, action, input_sentence=None):
#     # Map action to intent
#     intent_map = {
#         'start': 'set',
#         'set': 'set',
#         'check': 'check',
#         'play': 'play',
#         'verify': 'check',
#         'statusquery': 'check',
#         'on': 'set',
#         'off': 'set',
#         'up': 'set',
#         'down': 'set',
#         'pause': 'set',
#         'resume': 'set',
#         'stop': 'set',
#         'skip': 'set',
#         'loop': 'set',
#         'repeat': 'set',
#         'restart': 'set',
#         'add': 'set',
#         'clear': 'set',
#         'create': 'set',
#         'delete': 'set',
#         'shuffle': 'set',
#         'adjust': 'set',
#         'adjustspeed': 'set',
#         'resetspeed': 'set',
#         'addsong': 'set',
#         'removesong': 'set',
#         'reorder': 'set',
#         'selectplaylist': 'set',
#         'multiroom': 'set',
#         'switchdevice': 'set',
#         'volumesync': 'set',
#         'sync': 'set',
#         'turnon': 'set',
#         'turnoff': 'set',
#         'mute': 'set',
#         'unmute': 'set',
#         'adjustvolume': 'set',
#         'cancel': 'set',
#         'switchsource': 'set',
#         'enable': 'set',
#         'disable': 'set',
#     }
#     intent = intent_map.get(action.lower(), 'check')
    
#     # Extract entities based on keywords
#     entities = []
#     sentence_lower = sentence.lower()
#     input_lower = input_sentence.lower() if input_sentence else ""
#     entity_keywords = [
#         'ambient_music', 'status', 'sleep_timer', 'timer', 'voice_control',
#         'kitchen_lights', 'shutters', 'playlist', 'source', 'camera',
#         'main_entrance', 'volume', 'song', 'kitchen', 'bathroom', 'basement',
#         'attic', 'cellar', 'dining_room', 'library', 'living_room', 'toilet',
#         'outside', 'radio', 'podcast', 'queue', 'equalizer', 'all', 'none',
#         'playback', 'playbackspeed', 'artist', 'genre', 'multisource'
#     ]
#     for keyword in entity_keywords:
#         if (keyword in sentence_lower or keyword in input_lower) and keyword not in entities:
#             entities.append(keyword.replace(' ', '_'))
    
#     # Structured representation as a string
#     normalized = f"{intent}:{','.join(sorted(entities))}"
#     return normalized

# # Improved function to clean generated response
# def clean_response(text, tokenizer, input_sentence=None):
#     text = text.split(tokenizer.eos_token)[0].split("<|END|>")[0].strip()
#     # Remove echoed input sentence
#     if input_sentence and text.lower().startswith(input_sentence.lower()):
#         text = text[len(input_sentence):].strip()
#     # Remove 'Assistant:' or 'Instruction:' prefixes
#     text = re.sub(r'^(Assistant:|Instruction:)\s*', '', text, flags=re.IGNORECASE).strip()
#     text = re.sub(r'\s+([.,;:])', r'\1', text)
#     text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text).strip()
#     # Standardize synonyms
#     synonyms = {'reviewed': 'checked', 'retrieved': 'checked', 'active': 'checked', 'verified': 'checked'}
#     for old, new in synonyms.items():
#         text = text.replace(old, new)
#     text = text.split('.')[0].strip() + ('.' if not text.endswith('.') else '')
#     return text[:128]

# # Load model and tokenizer
# model_name = "./finetuned_llama/final_model"
# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
#     tokenizer.truncation_side = "right"
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#     )
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     print_gpu_memory("After Model Loading")
# except Exception as e:
#     print(f"Error loading model or tokenizer: {e}")
#     exit(1)

# # Load dataset
# try:
#     df = pd.read_csv("datasetBalanced1.csv")
#     print("Sample data:", df[['sentence', 'response']].head(3))
# except FileNotFoundError:
#     print("Error: datasetBalanced1.csv not found.")
#     exit(1)

# # Check for NaNs and handle stratification
# df['stratify_key'] = df['category'] + '_' + df['subcategory'] + '_' + df['action']
# if df['stratify_key'].isnull().any():
#     print("Warning: NaN values in stratify_key. Filling with 'unknown'.")
#     df['stratify_key'] = df['stratify_key'].fillna('unknown')

# try:
#     _, test_df = train_test_split(
#         df,
#         test_size=0.1,
#         stratify=df['stratify_key'],
#         random_state=42
#     )
# except ValueError as e:
#     print(f"Stratified split failed: {e}. Falling back to random split.")
#     _, test_df = train_test_split(df, test_size=0.1, random_state=42)

# print(f"Test size: {len(test_df)}")
# test_dataset = Dataset.from_pandas(test_df[['sentence', 'response', 'category', 'subcategory', 'action']])

# # Initialize scorers
# try:
#     sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
#     rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# except Exception as e:
#     print(f"Error initializing scorers: {e}")
#     exit(1)

# # Evaluation loop
# rouge_scores = defaultdict(list)
# bert_scores = defaultdict(list)
# sbert_scores = defaultdict(list)
# edit_distances = defaultdict(list)
# batch_size = 4
# sample_count = 0

# try:
#     open('non_zero_edit_distances.txt', 'w').close()
# except:
#     pass

# for i in range(0, len(test_dataset), batch_size):
#     indices = range(i, min(i + batch_size, len(test_dataset)))
#     batch = test_dataset.select(indices)
    
#     for example in batch:
#         try:
#             input_text = example['sentence']
#             expected_response = example['response']
#             category = example['category']
#             subcategory = example['subcategory']
#             action = example['action']
            
#             # Generate response
#             prompt = f"Instruction: {input_text}\nAssistant: "
#             inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=150,
#                     num_beams=4,
#                     do_sample=True,
#                     temperature=0.8,
#                     top_k=50,
#                     top_p=0.95,
#                     pad_token_id=tokenizer.eos_token_id,
#                 )
#             generated_response = clean_response(tokenizer.decode(outputs[0], skip_special_tokens=True), tokenizer, input_text)
            
#             # Normalize sentences
#             norm_expected = normalize_sentence(expected_response, action, input_text)
#             norm_generated = normalize_sentence(generated_response, action, input_text)
#             edit_dist = levenshtein_distance(norm_expected, norm_generated)
            
#             # Compute scores
#             rouge_score = rouge.score(expected_response, generated_response)['rougeL'].fmeasure
#             P, R, F1 = score([generated_response], [expected_response], lang="en", model_type="roberta-large", verbose=False)
#             bert_score = F1.item()
#             sbert_score = util.pytorch_cos_sim(
#                 sbert_model.encode(expected_response, convert_to_tensor=True),
#                 sbert_model.encode(generated_response, convert_to_tensor=True)
#             ).item()
            
#             # Store scores
#             combo = f"{category}_{subcategory}_{action}"
#             rouge_scores[combo].append(rouge_score)
#             bert_scores[combo].append(bert_score)
#             sbert_scores[combo].append(sbert_score)
#             edit_distances[combo].append(edit_dist)
            
#             # Print test result
#             sample_count += 1
#             print(f"Test Sample {sample_count}/{len(test_dataset)}:")
#             print(f"Sentence: {input_text}")
#             print(f"Expected Response: {expected_response}")
#             print(f"Generated Response: {generated_response}")
#             print(f"ROUGE-L Score: {rouge_score*100:.2f}%")
#             print(f"SBERT Cosine Score: {sbert_score*100:.2f}%")
#             print(f"Normalized Expected: {norm_expected}")
#             print(f"Normalized Generated: {norm_generated}")
#             print(f"Edit Distance: {edit_dist}")
#             print(f"BERTScore: {bert_score*100:.2f}%")
#             print("-" * 50 + "\n")
            
#             # Log non-zero edit distances
#             if edit_dist > 0:
#                 with open('non_zero_edit_distances.txt', 'a') as f:
#                     f.write(f"Test Sample {sample_count}/{len(test_dataset)}:\n")
#                     f.write(f"Sentence: {input_text}\n")
#                     f.write(f"Expected Response: {expected_response}\n")
#                     f.write(f"Generated Response: {generated_response}\n")
#                     f.write(f"Normalized Expected: {norm_expected}\n")
#                     f.write(f"Normalized Generated: {norm_generated}\n")
#                     f.write(f"Edit Distance: {edit_dist}\n")
#                     f.write(f"ROUGE-L Score: {rouge_score*100:.2f}%\n")
#                     f.write(f"SBERT Cosine Score: {sbert_score*100:.2f}%\n")
#                     f.write(f"BERTScore: {bert_score*100:.2f}%\n")
#                     f.write("-" * 50 + "\n")
            
#         except torch.cuda.OutOfMemoryError:
#             print(f"CUDA OOM Error for sentence: {input_text}")
#             clear_gpu_memory()
#             continue
#         except Exception as e:
#             print(f"Error for sentence {input_text}: {e}")
#             continue
    
#     clear_gpu_memory()

# # Save results
# results = []
# for combo in sorted(rouge_scores.keys()):
#     category, subcategory, action = combo.split('_')
#     results.append({
#         'Category': category,
#         'Subcategory': subcategory,
#         'Action': action,
#         'Count': len(rouge_scores[combo]),
#         'ROUGE-L (%)': round(np.mean(rouge_scores[combo]) * 100, 2),
#         'BERTScore (%)': round(np.mean(bert_scores[combo]) * 100, 2),
#         'SBERT Cosine (%)': round(np.mean(sbert_scores[combo]) * 100, 2),
#         'Edit Distance': round(np.mean(edit_distances[combo]), 2)
#     })

# df_metrics = pd.DataFrame(results)
# try:
#     df_metrics.to_csv('metrics_by_combination.csv', index=False)
#     print("\nResults saved to 'metrics_by_combination.csv'")
#     print(df_metrics.to_string(index=False))
# except Exception as e:
#     print(f"Error saving metrics: {e}")

# avg_rouge = np.mean([entry['ROUGE-L (%)'] for entry in results])
# avg_bert = np.mean([entry['BERTScore (%)'] for entry in results])
# avg_sbert = np.mean([entry['SBERT Cosine (%)'] for entry in results])
# avg_edit = np.mean([entry['Edit Distance'] for entry in results])

# print(f"\nAverage ROUGE-L: {avg_rouge:.2f}%")
# print(f"Average BERTScore: {avg_bert:.2f}%")
# print(f"Average SBERT Cosine: {avg_sbert:.2f}%")
# print(f"Average Edit Distance: {avg_edit:.2f}")

# clear_gpu_memory()