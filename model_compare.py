import os
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
try:
    import editdistance
except ImportError:
    editdistance = None
import time
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Define cache directory
CACHE_DIR = "./dataset_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Function to calculate directory size
def get_directory_size(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / 1e9  # Convert bytes to GiB

# Fallback Levenshtein distance implementation
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# Memory management function
def manage_gpu_memory(step_name: str, clear: bool = False) -> Dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if clear:
            torch.cuda.empty_cache()
            gc.collect()
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nMemory at {step_name}: Allocated={allocated:.2f}GiB, Reserved={reserved:.2f}GiB, Free={total-allocated:.2f}GiB")
        return {"allocated_gib": allocated, "reserved_gib": reserved, "free_gib": total - allocated}
    return {"allocated_gib": 0, "reserved_gib": 0, "free_gib": 0}

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

def normalize_sentence(sentence: str, category: Optional[str] = None, subcategory: Optional[str] = None, action: Optional[str] = None) -> Dict[str, str]:
    if nlp:
        doc = nlp(sentence.lower())
    else:
        doc = sentence.lower().split()
    
    intent = "unknown"
    device = category.lower() if category else "none"
    location = subcategory.lower() if subcategory else "none"
    action_slot = action.lower() if action else "none"
    
    if action_slot in intent_map:
        intent = intent_map[action_slot]
    else:
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

def representation_to_vector(rep: Dict[str, str]) -> np.ndarray:
    if nlp and spacy:
        intent_vec = nlp(rep["intent"]).vector
        action_vec = nlp(rep["action"]).vector
        device_vec = nlp(rep["device"]).vector
        location_vec = nlp(rep["location"]).vector
        return np.concatenate([intent_vec, action_vec, device_vec, location_vec])
    else:
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

def compute_cosine_similarity(rep1: Dict[str, str], rep2: Dict[str, str]) -> float:
    if spacy and cosine and nlp:
        vec1 = representation_to_vector(rep1)
        vec2 = representation_to_vector(rep2)
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        return 1 - cosine(vec1, vec2)
    else:
        vec1 = representation_to_vector(rep1)
        vec2 = representation_to_vector(rep2)
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0.0

# Load dataset
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
        test_size=0.1,  # Increased for more robust testing
        stratify=df['stratify_key'],
        random_state=42
    )
except ValueError:
    print("[INFO] Stratified split failed. Falling back to random split.")
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
columns_to_keep = ['sentence', 'response', 'category', 'subcategory', 'action']

# Create test dataset
try:
    test_dataset = Dataset.from_pandas(test_df[columns_to_keep]).map(
        lambda x: x,
        cache_file_name=os.path.join(CACHE_DIR, "test_cache.arrow")
    )
except Exception as e:
    print(f"Error caching dataset: {e}")
    print("Falling back to in-memory processing")
    test_dataset = Dataset.from_pandas(test_df[columns_to_keep])

# Clean response function
def clean_response(text: str, tokenizer: AutoTokenizer) -> str:
    text = text.split(tokenizer.eos_token)[0].split("<|END|>")[0].strip()
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    text = re.sub(r'^(Assistant\.|Instruction:).*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text).strip()
    text = text.replace("checked", "reviewed")
    text = text.split('.')[0].strip() + ('.' if not text.endswith('.') else '')
    return text[:128]

# Evaluation function
def evaluate_sample(
    input_text: str,
    expected_response: str,
    category: str,
    subcategory: str,
    action: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer
) -> Dict[str, float]:
    try:
        # Measure latency (tokenization + device transfer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        prompt = f"Instruction: {input_text}\nAssistant: "
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start_time
        
        # Measure memory before generation
        mem_before = manage_gpu_memory("Before Generation")
        
        # Measure first-token latency
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        first_token_start = time.perf_counter()
        with torch.no_grad():
            first_output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        first_token_latency = time.perf_counter() - first_token_start
        
        # Measure full generation time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_start_time = time.perf_counter()
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - gen_start_time
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_tokens = len(tokenizer.encode(generated_text))
        throughput = generated_tokens / gen_time if gen_time > 0 else 0
        
        # Measure memory after generation
        mem_after = manage_gpu_memory("After Generation", clear=True)
        
        # Decode response
        generated_response = clean_response(
            generated_text.split("Assistant:")[1].strip() if "Assistant:" in generated_text else generated_text,
            tokenizer
        )
        
        # Compute ROUGE-L
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_score = rouge.score(expected_response, generated_response)['rougeL'].fmeasure if expected_response and generated_response else 0.0
        
        # Compute cosine similarity
        expected_norm = normalize_sentence(expected_response, category, subcategory, action)
        generated_norm = normalize_sentence(generated_response, category, subcategory, action)
        cosine_sim = compute_cosine_similarity(expected_norm, generated_norm)
        
        # Compute edit distance
        if editdistance:
            ed = editdistance.eval(expected_response, generated_response)
        else:
            ed = levenshtein_distance(expected_response, generated_response)
        max_len = max(len(expected_response), len(generated_response), 1)
        edit_sim = 1.0 - (ed / max_len)
        
        # Compute intent accuracy
        intent_acc = 1.0 if generated_norm["intent"] == expected_norm["intent"] else 0.0
        
        return {
            "rouge_score": rouge_score,
            "cosine_sim": cosine_sim,
            "edit_sim": edit_sim,
            "latency_sec": latency,
            "first_token_latency_sec": first_token_latency,
            "gen_time_sec": gen_time,
            "throughput_tokens_per_sec": throughput,
            "mem_before_gib": mem_before["allocated_gib"],
            "mem_after_gib": mem_after["allocated_gib"],
            "generated_response": generated_response,
            "intent_accuracy": intent_acc
        }
    except Exception as e:
        print(f"Error evaluating sentence: {input_text}\nError: {e}")
        return {
            "rouge_score": 0.0,
            "cosine_sim": 0.0,
            "edit_sim": 0.0,
            "latency_sec": 0.0,
            "first_token_latency_sec": 0.0,
            "gen_time_sec": 0.0,
            "throughput_tokens_per_sec": 0.0,
            "mem_before_gib": 0.0,
            "mem_after_gib": 0.0,
            "generated_response": "",
            "intent_accuracy": 0.0
        }

# Function to evaluate a model
def evaluate_model(model_path: str, model_name: str, test_dataset: Dataset, quantize: bool = False) -> List[Dict]:
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        
        # Configure quantization if requested
        quantization_config = None
        if quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if not quantize else None,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            device_map="auto"
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        manage_gpu_memory(f"After Loading {model_name}")
        
        # Measure model size
        model_size_gib = get_directory_size(model_path)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return []
    
    # Evaluate test performance
    test_dataset_with_metadata = Dataset.from_pandas(test_df[['sentence', 'response', 'category', 'subcategory', 'action']])
    test_dataloader = DataLoader(test_dataset_with_metadata, batch_size=1, shuffle=False)  # Reduced batch size
    results = []
    
    for batch in tqdm(test_dataloader, desc=f"Evaluating {model_name}"):
        input_texts = batch['sentence']
        expected_responses = batch['response']
        categories = batch['category']
        subcategories = batch['subcategory']
        actions = batch['action']
        
        try:
            for i, (input_text, expected, category, subcategory, action) in enumerate(zip(
                input_texts, expected_responses, categories, subcategories, actions
            )):
                metrics = evaluate_sample(
                    input_text, expected, category, subcategory, action, model, tokenizer
                )
                combination = f"{category}_{subcategory}_{action}"
                result = {
                    "Model": model_name,
                    "Category": category,
                    "Subcategory": subcategory,
                    "Action": action,
                    "Sentence": input_text,
                    "Expected": expected,
                    "Generated": metrics["generated_response"],
                    "ROUGE-L (%)": metrics["rouge_score"] * 100,
                    "Cosine Similarity (%)": metrics["cosine_sim"] * 100,
                    "Edit Distance Similarity (%)": metrics["edit_sim"] * 100,
                    "Latency (sec)": metrics["latency_sec"],
                    "First-Token Latency (sec)": metrics["first_token_latency_sec"],
                    "Generation Time (sec)": metrics["gen_time_sec"],
                    "Throughput (tokens/sec)": metrics["throughput_tokens_per_sec"],
                    "Memory Before (GiB)": metrics["mem_before_gib"],
                    "Memory After (GiB)": metrics["mem_after_gib"],
                    "Model Size (GiB)": model_size_gib,
                    "Intent Accuracy (%)": metrics["intent_accuracy"] * 100
                }
                results.append(result)
                
                print(f"Model: {model_name}")
                print(f"Sentence: {input_text}")
                print(f"Expected: {expected}")
                print(f"Generated: {metrics['generated_response']}")
                print(f"ROUGE-L: {metrics['rouge_score']*100:.2f}%")
                print(f"Cosine Similarity: {metrics['cosine_sim']*100:.2f}%")
                print(f"Edit Distance Similarity: {metrics['edit_sim']*100:.2f}%")
                print(f"Intent Accuracy: {metrics['intent_accuracy']*100:.2f}%")
                print(f"Latency: {metrics['latency_sec']:.6f} sec")
                print(f"First-Token Latency: {metrics['first_token_latency_sec']:.6f} sec")
                print(f"Generation Time: {metrics['gen_time_sec']:.4f} sec")
                print(f"Throughput: {metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
                print(f"Memory Before: {metrics['mem_before_gib']:.2f} GiB")
                print(f"Memory After: {metrics['mem_after_gib']:.2f} GiB")
                print(f"Model Size: {model_size_gib:.2f} GiB")
                print(f"Processed {len(results)}/{len(test_dataset_with_metadata)} examples\n")
            
            manage_gpu_memory(f"After batch evaluation for {model_name}", clear=True)
        except Exception as e:
            print(f"Error processing batch for {model_name}: {e}")
            continue
    
    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

# Save metrics
def save_metrics(results: List[Dict], filename: str):
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

# Evaluate both models
models_to_evaluate = [
    {"path": "./finetuned_llama321_merged", "name": "Llama-3.2-1B-Merged", "quantize": False},
    {"path": "./finetuned_llama321_4bit", "name": "Llama-3.2-1B-4bit", "quantize": True}
]

all_results = []
for model_info in models_to_evaluate:
    results = evaluate_model(
        model_info["path"],
        model_info["name"],
        test_dataset,
        quantize=model_info["quantize"]
    )
    all_results.extend(results)

# Aggregate metrics
if all_results:
    df_results = pd.DataFrame(all_results)
    summary = df_results.groupby("Model").agg({
        "ROUGE-L (%)": ["mean", "std"],
        "Cosine Similarity (%)": ["mean", "std"],
        "Edit Distance Similarity (%)": ["mean", "std"],
        "Intent Accuracy (%)": ["mean", "std"],
        "Latency (sec)": ["mean", "std"],
        "First-Token Latency (sec)": ["mean", "std"],
        "Generation Time (sec)": ["mean", "std"],
        "Throughput (tokens/sec)": ["mean", "std"],
        "Memory Before (GiB)": ["mean", "max"],
        "Memory After (GiB)": ["mean", "max"],
        "Model Size (GiB)": ["mean"]
    }).round(4)
    
    print("\nPerformance Comparison Summary:")
    print(summary.to_string())
    save_metrics(all_results, "model_comparison_metrics.csv")
    save_metrics(summary.reset_index().to_dict('records'), "model_comparison_summary.csv")
else:
    print("\nNo results to save. Evaluation loop produced no metrics.")

manage_gpu_memory("After All Evaluations", clear=True)
