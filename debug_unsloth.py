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
import re
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Function to check GPU memory usage
def print_gpu_memory(step_name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"\nMemory Usage at {step_name}:")
        logger.info(f"  Allocated: {allocated:.2f} GiB")
        logger.info(f"  Reserved: {reserved:.2f} GiB")
        logger.info(f"  Free: {total - allocated:.2f} GiB")
        logger.info(f"  Total: {total:.2f} GiB")

# Clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_memory("After Clearing GPU Memory")

clear_gpu_memory()

# 1. Load and preprocess CSV
data = "./datasetNew.csv"  # Update to your actual dataset path
required_columns = ['Sentence', 'Response']
try:
    logger.info("Loading CSV...")
    df = pd.read_csv(data)
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Dataset must contain {required_columns}. Found: {df.columns.tolist()}")
        raise ValueError("Missing required columns.")
except FileNotFoundError:
    logger.error(f"{data} not found.")
    raise
except Exception as e:
    logger.error(f"Error loading CSV: {e}")
    raise

# Clean responses to remove timestamps and special tokens
def clean_response(text):
    """Clean response to remove timestamps and special tokens."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.?\d*', '', text)  # Remove timestamps
    text = re.sub(r'<\|\w+\|>', '', text)  # Remove special tokens like <|END|>
    text = re.sub(r'(Month|Day|Year|Hour|Minute|Second|Weekday|Temperature|Humidity|.*\bAssistant\b.*$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip('.,:; ')  # Normalize whitespace and punctuation
    return text

df['Response'] = df['Response'].apply(clean_response)

# Create input and output for training
df['input'] = df['Sentence']
df['output'] = df['Response']

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df[['input', 'output']])
test_dataset = Dataset.from_pandas(test_df[['input', 'output']])

# 2. Load model and tokenizer
model_name = "./Llama-3.2-3B"  # Update to a valid model path or Hugging Face model name
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
    logger.error(f"Error loading model: {e}")
    raise

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# 3. Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print_gpu_memory("After Applying LoRA")

# 4. Preprocess dataset
def preprocess_function(examples):
    inputs = [f"Instruction: {inp} \nAssistant: {out} <|END|>" for inp, out in zip(examples['input'], examples['output'])]
    tokenized = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['input', 'output'])
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=['input', 'output'])

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_llama",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=1e-4,
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
    logger.error(f"CUDA OOM Error during training: {e}")
    clear_gpu_memory()
    raise
except Exception as e:
    logger.error(f"Error during training: {e}")
    raise

# 8. Save the fine-tuned model
model.save_pretrained("./finetuned_Wcsa")
tokenizer.save_pretrained("./finetuned_Wcsa")

# 9. Evaluate test accuracy
model.eval()
bleu_scores = []
rouge_scores = []
exact_matches = []
rouge_scorer_inst = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoother = SmoothingFunction().method1

for example in test_dataset:
    input_ids = example['input_ids']
    expected_response = example.get('output', '')  # Fallback if 'output' not in dataset
    prompt = tokenizer.decode(input_ids, skip_special_tokens=True).split("Assistant:")[0].strip() + "Assistant: "
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
            no_repeat_ngram_size=3,
            do_sample=False,
            temperature=0.6
        )
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
        generated_response = clean_response(generated_response)
        expected_response = clean_response(expected_response)
        bleu_score = sentence_bleu([expected_response.split()], generated_response.split(), smoothing_function=smoother)
        rouge_score = rouge_scorer_inst.score(expected_response, generated_response)['rougeL'].fmeasure
        exact_match = 1 if generated_response.lower() == expected_response.lower() else 0
        bleu_scores.append(bleu_score)
        rouge_scores.append(rouge_score)
        exact_matches.append(exact_match)
        logger.info(f"Input: {prompt}")
        logger.info(f"Expected: {expected_response}")
        logger.info(f"Generated: {generated_response}")
        logger.info(f"BLEU Score: {bleu_score:.4f}")
        logger.info(f"ROUGE-L Score: {rouge_score:.4f}")
        logger.info(f"Exact Match: {exact_match}\n")
        del inputs, outputs
        clear_gpu_memory()
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM Error during generation: {e}")
        clear_gpu_memory()
        continue
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        continue

# 10. Calculate and display overall metrics
avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
avg_exact = np.mean(exact_matches) if exact_matches else 0.0
logger.info(f"\nOverall Average BLEU Score: {avg_bleu:.4f}")
logger.info(f"Overall Average ROUGE-L Score: {avg_rouge:.4f}")
logger.info(f"Overall Exact Match Accuracy: {avg_exact:.4f}")

# Clean up
clear_gpu_memory()


# import os
# import logging
# import pandas as pd
# import torch
# from unsloth import FastLanguageModel, is_bfloat16_supported
# from trl import SFTTrainer
# from transformers import TrainingArguments, TrainerCallback
# from datasets import Dataset
# import transformers

# # -------------------------------
# # Step 1: Disable Unsloth optimizations
# # -------------------------------
# os.environ["UNSLOTH_DISABLE_FUSED_CE"] = "1"
# os.environ["UNSLOTH_DISABLE_ALL_OPTIMIZATIONS"] = "1"

# # -------------------------------
# # Step 2: Setup logging
# # -------------------------------
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # -------------------------------
# # Step 3: Verify environment
# # -------------------------------
# try:
#     logger.info(f"PyTorch version: {torch.__version__}")
#     logger.info(f"Transformers version: {transformers.__version__}")
#     logger.info(f"CUDA available: {torch.cuda.is_available()}")
#     logger.info(f"CUDA device count: {torch.cuda.device_count()}")
#     logger.info(f"Current device: {torch.cuda.get_device_name(0)}")
# except Exception as e:
#     logger.error(f"Error checking environment: {str(e)}")
#     raise

# # -------------------------------
# # Step 4: Load model & tokenizer
# # -------------------------------
# max_seq_length = 512  # Reduced to save VRAM
# dtype = None
# load_in_4bit = True

# try:
#     logger.info("Loading model and tokenizer...")
#     model_name = "./model"  # Local path to unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=model_name,
#         max_seq_length=max_seq_length,
#         dtype=dtype,
#         load_in_4bit=load_in_4bit,
#         device_map={"": 0},  # Use GPU 0
#     )
#     logger.info(f"EOS token: {tokenizer.eos_token}")
#     logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
# except Exception as e:
#     logger.error(f"Error loading model/tokenizer: {str(e)}")
#     raise

# # -------------------------------
# # Step 5: Load and preprocess CSV
# # -------------------------------
# try:
#     logger.info("Loading CSV...")
#     df = pd.read_csv("/home/mcw/Aditya/unsloth/music.csv")
#     required_columns = ['Sentence', 'Response']
#     if not all(col in df.columns for col in required_columns):
#         raise ValueError("CSV must contain 'Sentence' and 'Response' columns")
    
#     # Drop nulls
#     df = df.dropna(subset=['Sentence', 'Response'])
    
#     # Filter for lighting-related examples
#     lighting_keywords = ['light', 'lights', 'kitchen', 'bedroom', 'living', 'room', 'lamp', 'illuminate', 'turn on', 'turn off']
#     df = df[df['Sentence'].str.lower().str.contains('|'.join(lighting_keywords), na=False)]
#     logger.info(f"Filtered {len(df)} lighting examples")
#     logger.info(f"Sample rows:\n{df.head(3).to_dict(orient='records')}")
#     logger.info(f"CSV info:\n{df.info()}")
#     logger.info(f"Missing values:\n{df.isnull().sum()}")
# except Exception as e:
#     logger.error(f"Error loading CSV: {str(e)}")
#     raise

# # Prompt format with delimiters
# prompt_template = "### Input: {}\n### Output: {}"

# def formatting_prompts_func(examples):
#     inputs = examples["Sentence"]
#     outputs = examples["Response"]
#     texts = []
#     for input, output in zip(inputs, outputs):
#         text = prompt_template.format(input, output) + tokenizer.eos_token
#         texts.append(text)
#     return {"text": texts}

# try:
#     logger.info("Creating dataset...")
#     dataset = Dataset.from_pandas(df)
#     # Split into train and validation
#     dataset = dataset.train_test_split(test_size=0.1, seed=3407)
#     train_dataset = dataset["train"]
#     eval_dataset = dataset["test"]
#     train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
#     eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
#     logger.info(f"Train dataset size: {len(train_dataset)}")
#     logger.info(f"Validation dataset size: {len(eval_dataset)}")
#     logger.info(f"Sample train data: {train_dataset[0]}")
    
#     # Log tokenized sample
#     sample_prompt = train_dataset[0]["text"]
#     tokenized = tokenizer(sample_prompt, return_tensors="pt")
#     logger.info(f"Sample tokenized input IDs: {tokenized.input_ids[0].tolist()[:50]}...")
#     logger.info(f"Sample decoded tokens: {tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)}")
# except Exception as e:
#     logger.error(f"Error creating dataset: {str(e)}")
#     raise

# # -------------------------------
# # Step 6: Apply LoRA
# # -------------------------------
# try:
#     logger.info("Applying LoRA...")
#     model = FastLanguageModel.get_peft_model(
#         model,
#         r=64,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#         lora_alpha=64,
#         lora_dropout=0.1,
#         bias="none",
#         use_gradient_checkpointing="unsloth",
#         random_state=3407,
#         use_rslora=False,
#         loftq_config=None,
#     )
# except Exception as e:
#     logger.error(f"Error applying LoRA: {str(e)}")
#     raise

# # -------------------------------
# # Step 7: TrainingArguments
# # -----------------------
# try:
#     logger.info("Setting up training arguments...")
#     training_args = TrainingArguments(
#         per_device_train_batch_size=4,
#         gradient_accumulation_steps=1,
#         warmup_steps=5,
#         num_train_epochs=3,
#         learning_rate=1e-4,
#         fp16=not is_bfloat16_supported(),
#         bf16=is_bfloat16_supported(),
#         logging_steps=10,
#         optim="adamw_8bit",
#         weight_decay=0.1,
#         lr_scheduler_type="linear",
#         seed=3407,
#         output_dir="outputs",
#         report_to="none",
#         gradient_checkpointing=True,
#         eval_strategy="steps",
#         eval_steps=20,
#         save_strategy="steps",
#         save_steps=20,
#         load_best_model_at_end=True,
#         metric_for_best_model="loss",
#     )
# except Exception as e:
#     logger.error(f"Error setting up training arguments: {str(e)}")
#     raise

# # -------------------------------
# # Step 8: Custom callback for logging
# # -------------------------------
# class DataLoggingCallback(TrainerCallback):
#     def on_train_begin(self, args, state, control, **kwargs):
#         logger.info("Training started")
#         logger.info("Skipping first batch logging due to callback limitations")

# # -------------------------------
# # Step 9: Train the model
# # -------------------------------
# try:
#     logger.info("Starting training...")
#     trainer = SFTTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         dataset_text_field="text",
#         max_seq_length=max_seq_length,
#         dataset_num_proc=2,
#         packing=False,
#         args=training_args,
#         callbacks=[DataLoggingCallback()],
#     )
    
#     # Log sample prediction
#     def log_sample_prediction(stage):
#         sample_input = train_dataset[0]["Sentence"]
#         prompt = prompt_template.format(sample_input, "") + tokenizer.eos_token
#         inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
#         outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
#         decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         response = decoded.split("### Output:")[-1].strip() if "### Output:" in decoded else decoded.strip()
#         logger.info(f"{stage} prediction: Input: {sample_input}\nOutput: {response}")
    
#     logger.info("Pre-training sample prediction...")
#     log_sample_prediction("Pre-training")
    
#     trainer.train()
    
#     logger.info("Post-training sample prediction...")
#     log_sample_prediction("Post-training")
    
#     # Log first training batch
#     logger.info("Logging first training batch post-training...")
#     dataloader = trainer.get_train_dataloader()
#     batch = next(iter(dataloader))
#     input_ids = batch["input_ids"][0].tolist()[:50]
#     logger.info(f"Post-training batch input IDs: {input_ids}...")
#     logger.info(f"Post-training batch decoded: {tokenizer.decode(input_ids, skip_special_tokens=True)}")
# except Exception as e:
#     logger.error(f"Error during training: {str(e)}")
#     raise

# # -------------------------------
# # Step 10: Save model
# # -------------------------------
# try:
#     logger.info("Saving model...")
#     model.save_pretrained("smart-home-lora-llama")
#     tokenizer.save_pretrained("smart-home-lora-llama")
# except Exception as e:
#     logger.error(f"Error saving model: {str(e)}")
#     raise

# # -------------------------------
# # Step 11: Test the model
# # -------------------------------
# try:
#     logger.info("Testing model...")
#     def generate_response(question, max_new_tokens=50):
#         prompt = prompt_template.format(question, "")  # <-- Remove eos_token here!
#         logger.info(f"Test prompt: {prompt}")
#         inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0" if torch.cuda.is_available() else "cpu")
#         model.to(inputs.input_ids.device)
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#         )
#         logger.info(f"Raw output tokens: {outputs[0].tolist()}")
#         logger.info(f"Token-to-text: {tokenizer.convert_ids_to_tokens(outputs[0].tolist())}")
#         decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logger.info(f"Decoded output: {decoded}")
#         response = decoded.split("### Output:")[-1].strip() if "### Output:" in decoded else decoded.strip()
#         return response

#     # Test 1: Lighting query
#     test_question = "Turn on the music"
#     response = generate_response(test_question)
#     logger.info(f"💡 Test Question 1: {test_question}\n Model Response 1: {response}")

#     # Test 2: Dataset lighting example
#     test_question_2 = train_dataset[0]["Sentence"]
#     response_2 = generate_response(test_question_2)
#     logger.info(f"💡 Test Question 2: {test_question_2}\n Model Response 2: {response_2}")
# except Exception as e:
#     logger.error(f"Error testing model: {str(e)}")
#     raise