
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from bert_score import score
import gc
from collections import defaultdict
import numpy as np
import re

# ========== Memory Cleanup Helpers ==========
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

# ========== Load Dataset ==========
data_path = "./datasetBalanced1.csv"
df = pd.read_csv(data_path)

required_columns = ['Category', 'Subcategory', 'Action', 'Sentence', 'Response']
assert all(col in df.columns for col in required_columns), f"Missing columns: {required_columns}"

df['Category'] = df['Category'].str.lower()
df['Subcategory'] = df['Subcategory'].str.lower()
df['stratify_key'] = df['Category'] + '_' + df['Subcategory'] + '_' + df['Action']
combination_counts = df['stratify_key'].value_counts()
valid_combinations = combination_counts[combination_counts >= 2].index
df = df[df['stratify_key'].isin(valid_combinations)]

df['input'] = df.apply(lambda row: f"Category: {row['Category']}, Subcategory: {row['Subcategory']}, Action: {row['Action']}, Sentence: {row['Sentence']}", axis=1)
df['output'] = df['Response']

try:
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['stratify_key'])
except ValueError:
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# ========== Load Fine-tuned Model ==========
model_path = "./finetuned_llama2"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_state_dict=True,
)

model.gradient_checkpointing_enable()
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

print_gpu_memory("After Loading Fine-tuned Model")

# ========== Prepare Test Set ==========
test_dataset_with_metadata = Dataset.from_pandas(test_df[['input', 'output', 'Category', 'Subcategory', 'Action']])

def clean_response(text):
    text = text.split(tokenizer.eos_token)[0].split("<|END|>")[0].strip()
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    text = re.sub(r'^(Assistant\.|Instruction:).*', '', text, flags=re.IGNORECASE).strip()
    return text[:256]

# ========== Evaluate ==========
rouge_scores = defaultdict(list)
bert_scores = defaultdict(list)
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

for example in test_dataset_with_metadata:
    input_text = example['input']
    expected_response = example['output']
    category = example['Category']
    subcategory = example['Subcategory']
    action = example['Action']
    prompt = f"Instruction: {input_text}\nAssistant: "

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
        print(generated_response)
        generated_response = clean_response(generated_response)

        # Compute ROUGE
        if expected_response is None or generated_response is None:
            rouge_score = 0.0
            bert_score_val = 0.0
        else:
            rouge_score = rouge.score(expected_response, generated_response)['rougeL'].fmeasure

        # Compute BERTScore safely
        try:
            P, R, F1 = score([generated_response], [expected_response], lang="en", verbose=False)
            bert_score_val = F1.item()
        except Exception as e:
            print(f"BERTScore error for input:\n{input_text}\nError: {e}")
            bert_score_val = 0.0

        # Save scores by group
        combination = f"{category}_{subcategory}_{action}"
        rouge_scores[combination].append(rouge_score)
        bert_scores[combination].append(bert_score_val)

        print(f"Input: {input_text}")
        print(f"Expected: {expected_response}")
        print(f"Generated: {generated_response}")
        print(f"ROUGE-L Score: {rouge_score*100:.2f}%")
        print(f"BERTScore: {bert_score_val*100:.2f}%\n")

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error: {e}")
        clear_gpu_memory()
        continue

# ========== Aggregate Results ==========
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

results_df = pd.DataFrame(results)
print("\nMetrics by Category, Subcategory, and Action:")
print(results_df.to_string(index=False))
results_df.to_csv('metrics_by_combination.csv', index=False)
print("\nResults saved to 'metrics_by_combination.csv'")

clear_gpu_memory()

# import pandas as pd

# # Load the dataset
# df = pd.read_csv('datasetBalanced1.csv')

# # #change column to lower
# # # for col in ['Category', 'Subcategory', 'Action']:
# # #     df[col] = df[col].str.strip().str.lower()

# # # # If you want to save the cleaned CSV (with Sentence, Response preserved)
# # # df.to_csv("datasetBalanced1.csv", index=False)

# # # import peft
# # # print(peft.__version__)
# # # import sys
# # # print("Python interpreter:", sys.executable)

# # # #import peft  # Try again



# # # Ensure 'Count' column is numeric
# # # df['Count'] = pd.to_numeric(df['Count'], errors='coerce').fillna(0).astype(int)

# # # Clean columns: strip spaces and convert to lowercase
# # # for col in ['Category', 'Subcategory', 'Action']:
# # #     df[col] = df[col].str.strip().str.lower()

# # Count unique combinations
# count_df = df.groupby(['Category', 'Subcategory', 'Action']).size().reset_index(name='Count')

# # Optionally sort the result
# count_df = count_df.sort_values(by=['Category', 'Subcategory', 'Action']).reset_index(drop=True)

# # Save to CSV
# count_df.to_csv("count_output.csv", index=False)

# # # Find rows where Action is None or NaN
# # #none_actions = df[df['Action']=='none']




# # # Define filter conditions for each required combinatio

# # # # List your filter combinations as tuples
# # # filters = [
# # #     ('camera', 'none', 'no_action'),
# # #     ('camera', 'none', 'on'),
# # #     ('camera', 'outside', 'no_action'),
# # #     ('camera', 'outside', 'on'),
# # #     ('camera', 'toilet', 'on'),
# # #     ('lights', 'all', 'on'),
# # #     ('lights', 'attic', 'off'),
# # #     ('lights', 'attic', 'on'),
# # #     ('lights', 'basement', 'off'),
# # #     ('lights', 'basement', 'on'),
# # #     ('lights', 'bathroom', 'off'),
# # #     ('lights', 'bathroom', 'on'),
# # #     ('lights', 'cellar', 'off'),
# # #     ('lights', 'cellar', 'on'),
# # #     ('lights', 'dining room', 'no_action'),

# # # ]

# # # # Build a boolean mask for all the filter combinations
# # # mask = pd.Series(False, index=df.index)
# # # for cat, subcat, action in filters:
# # #     mask |= (
# # #         (df['Category'].str.lower() == cat.lower()) &
# # #         (df['Subcategory'].str.lower() == subcat.lower()) &
# # #         (df['Action'].str.lower() == action.lower())
# # #     )

# # # # Filter the DataFrame
# # # filtered_df = df[mask]

# # # # Optional: Show all rows and columns in terminal
# # # pd.set_option('display.max_rows', None)
# # # pd.set_option('display.max_columns', None)
# # # pd.set_option('display.width', None)
# # # pd.set_option('display.max_colwidth', None)

# # # print(filtered_df)



# # # Replace 'none' with 'no_action' in the 'Action' column
# # # df['Action'] = df['Action'].replace('none', 'no_action')


# # # Drop 'Category', 'Action', and 'Subcategory' columns
# # # df = df.drop(columns=['Category','Action','Subcategory'])
# # # df.to_csv("datasetNew.csv", index=False)


# # # # Save the updated DataFrame back to the CSV
# # # df.to_csv("datasetBalanced1.csv", index=False)

# # #nan_rows = df[df.isna().any(axis=1)]

# # # Print the rows with NaNs
# # #print(nan_rows)

# # # import torch
# # # from accelerate import Accelerator


# # # accelerator = Accelerator()
# # # print(accelerator.state)
# # # print("CUDA available:", torch.cuda.is_available())
# # # print("Device count:", torch.cuda.device_count())
# # # print("Devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# # # from transformers import AutoTokenizer
# # # import pandas as pd
# # # tokenizer = AutoTokenizer.from_pretrained("./Llama-3.2-3B")
# # # df = pd.read_csv("./datasetBalanced1.csv")
# # # df['input'] = df.apply(lambda row: f"Category: {row['Category']}, Subcategory: {row['Subcategory']}, Action: {row['Action']}, Sentence: {row['Sentence']}", axis=1)
# # # df['output'] = df['Response']
# # # tokenized_lengths = [len(tokenizer.encode(text)) for text in df['input'].tolist() + df['output'].tolist()]
# # # print("Max tokens:", max(tokenized_lengths))



# # import os
# # import pandas as pd
# # import torch
# # from transformers import AutoModelForCausalLM, AutoTokenizer
# # from datasets import Dataset
# # from sklearn.model_selection import train_test_split
# # from rouge_score import rouge_scorer
# # from bert_score import score
# # import gc
# # from collections import defaultdict
# # import numpy as np
# # import re

# # # === Memory Cleanup ===
# # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# # def print_gpu_memory(step_name):
# #     if torch.cuda.is_available():
# #         torch.cuda.synchronize()
# #         allocated = torch.cuda.memory_allocated(0) / 1e9
# #         reserved = torch.cuda.memory_reserved(0) / 1e9
# #         total = torch.cuda.get_device_properties(0).total_memory / 1e9
# #         print(f"\nMemory Usage at {step_name}:\n  Allocated: {allocated:.2f} GiB\n  Reserved: {reserved:.2f} GiB\n  Free: {total - allocated:.2f} GiB\n  Total: {total:.2f} GiB")

# # def clear_gpu_memory():
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #         gc.collect()
# #         print_gpu_memory("After Clearing GPU Memory")

# # clear_gpu_memory()

# # # === Load Dataset ===
# # data_path = "./datasetBalanced1.csv"
# # df = pd.read_csv(data_path)

# # required_columns = ['Category', 'Subcategory', 'Action', 'Sentence', 'Response']
# # assert all(col in df.columns for col in required_columns), "Missing required columns"

# # df['Category'] = df['Category'].str.lower()
# # df['Subcategory'] = df['Subcategory'].str.lower()
# # df['stratify_key'] = df['Category'] + '_' + df['Subcategory'] + '_' + df['Action']
# # combination_counts = df['stratify_key'].value_counts()
# # valid_combinations = combination_counts[combination_counts >= 2].index
# # df = df[df['stratify_key'].isin(valid_combinations)]

# # df['input'] = df.apply(lambda row: f"Category: {row['Category']}, Subcategory: {row['Subcategory']}, Action: {row['Action']}, Sentence: {row['Sentence']}", axis=1)
# # df['output'] = df['Response']

# # _, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['stratify_key'])
# # print(f"Test set size: {len(test_df)}")
# # test_dataset = Dataset.from_pandas(test_df[['input', 'output', 'Category', 'Subcategory', 'Action']])

# # # === Load Fine-tuned Model ===
# # model_path = "./finetuned_llama2"
# # tokenizer = AutoTokenizer.from_pretrained(model_path)
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_path,
# #     torch_dtype=torch.bfloat16,
# #     device_map="auto",
# #     low_cpu_mem_usage=True
# # )

# # tokenizer.pad_token = tokenizer.eos_token
# # model.config.pad_token_id = tokenizer.eos_token_id
# # model.eval()
# # print_gpu_memory("After Loading Fine-Tuned Model")

# # # === Evaluation Functions ===
# # rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# # rouge_scores = defaultdict(list)
# # bert_scores = defaultdict(list)

# # def clean_response(text):
# #     text = text.split("<|END|>")[0].strip()
# #     text = re.sub(r'(?<!\d)\s+', ' ', text).strip('.,:; ')
# #     text = re.sub(r'^(Assistant\.|Instruction:).*', '', text, flags=re.IGNORECASE).strip()
# #     return text

# # def post_process_response(generated, category, subcategory, action):
# #     if category == 'music' and action == 'volume_sync':
# #         return f"Volume synced for {subcategory}."
# #     elif category == 'camera' and 'at' in generated.lower():
# #         time_match = re.search(r'\d{1,2}:\d{2}\s?(AM|PM|am|pm)', generated)
# #         if time_match:
# #             return f"Camera will start at {time_match.group(0)}."
# #     elif category == 'lights' and action == 'no_action':
# #         return f"{subcategory.capitalize()} lighting status shown."
# #     elif category == 'music' and action == 'play' and subcategory == 'genre':
# #         genre_match = re.search(r'\b(jazz|classical|pop|rock|reggae|hip-hop|blues|country|indie|electronic)\b', generated, re.IGNORECASE)
# #         if genre_match:
# #             return f"{genre_match.group(0).capitalize()} music playing."
# #     return generated

# # # === Inference Loop ===
# # for example in test_dataset:
# #     input_text = example['input']
# #     expected_response = example['output']
# #     category = example['Category']
# #     subcategory = example['Subcategory']
# #     action = example['Action']
# #     prompt = f"Instruction: {input_text}\nAssistant: "

# #     try:
# #         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
# #         outputs = model.generate(
# #             **inputs,
# #             max_new_tokens=50,
# #             num_beams=3,
# #             do_sample=True,
# #             temperature=0.7,
# #             top_p=0.9,
# #             no_repeat_ngram_size=3,
# #         )
# #         generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
# #         generated_response = clean_response(generated_response)
# #         generated_response = post_process_response(generated_response, category, subcategory, action)

# #         # Evaluate
# #         if expected_response and generated_response:
# #             rouge_score = rouge_scorer.score(expected_response, generated_response)['rougeL'].fmeasure
# #             P, R, F1 = score([generated_response], [expected_response], lang="en", verbose=False)
# #             bert_score = F1.item()
# #             key = f"{category}_{subcategory}_{action}"
# #             rouge_scores[key].append(rouge_score)
# #             bert_scores[key].append(bert_score)

# #             print(f"Input: {input_text}")
# #             print(f"Expected: {expected_response}")
# #             print(f"Generated: {generated_response}")
# #             print(f"ROUGE-L: {rouge_score*100:.2f}% | BERTScore: {bert_score*100:.2f}%\n")
# #     except torch.cuda.OutOfMemoryError as e:
# #         print(f"CUDA OOM Error: {e}")
# #         clear_gpu_memory()
# #         continue

# # # === Aggregate and Save Results ===
# # results = []
# # for combo in sorted(rouge_scores.keys()):
# #     rouge_avg = np.mean(rouge_scores[combo]) * 100
# #     bert_avg = np.mean(bert_scores[combo]) * 100
# #     category, subcategory, action = combo.split('_',2)
# #     results.append({
# #         'Category': category,
# #         'Subcategory': subcategory,
# #         'Action': action,
# #         'Count': len(rouge_scores[combo]),
# #         'ROUGE-L (%)': round(rouge_avg, 2),
# #         'BERTScore (%)': round(bert_avg, 2)
# #     })

# # results_df = pd.DataFrame(results)
# # print("\nEvaluation Metrics by (Category, Subcategory, Action):")
# # print(results_df.to_string(index=False))
# # results_df.to_csv('metrics_by_combination.csv', index=False)
# # print("\nResults saved to 'metrics_by_combination.csv'")

# # # === Cleanup ===
# # clear_gpu_memory()

