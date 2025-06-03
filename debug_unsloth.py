import os
import logging
import pandas as pd
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import Dataset
import transformers

# -------------------------------
# Step 1: Disable Unsloth optimizations
# -------------------------------
os.environ["UNSLOTH_DISABLE_FUSED_CE"] = "1"
os.environ["UNSLOTH_DISABLE_ALL_OPTIMIZATIONS"] = "1"

# -------------------------------
# Step 2: Setup logging
# -------------------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------
# Step 3: Verify environment
# -------------------------------
try:
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    logger.error(f"Error checking environment: {str(e)}")
    raise

# -------------------------------
# Step 4: Load model & tokenizer
# -------------------------------
max_seq_length = 512  # Reduced to save VRAM
dtype = None
load_in_4bit = True

try:
    logger.info("Loading model and tokenizer...")
    model_name = "./model"  # Local path to unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map={"": 0},  # Use GPU 0
    )
    logger.info(f"EOS token: {tokenizer.eos_token}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
except Exception as e:
    logger.error(f"Error loading model/tokenizer: {str(e)}")
    raise

# -------------------------------
# Step 5: Load and preprocess CSV
# -------------------------------
try:
    logger.info("Loading CSV...")
    df = pd.read_csv("/home/mcw/Aditya/unsloth/music.csv")
    required_columns = ['Sentence', 'Response']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV must contain 'Sentence' and 'Response' columns")
    
    # Drop nulls
    df = df.dropna(subset=['Sentence', 'Response'])
    
    # Filter for lighting-related examples
    lighting_keywords = ['light', 'lights', 'kitchen', 'bedroom', 'living', 'room', 'lamp', 'illuminate', 'turn on', 'turn off']
    df = df[df['Sentence'].str.lower().str.contains('|'.join(lighting_keywords), na=False)]
    logger.info(f"Filtered {len(df)} lighting examples")
    logger.info(f"Sample rows:\n{df.head(3).to_dict(orient='records')}")
    logger.info(f"CSV info:\n{df.info()}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
except Exception as e:
    logger.error(f"Error loading CSV: {str(e)}")
    raise

# Prompt format with delimiters
prompt_template = "### Input: {}\n### Output: {}"

def formatting_prompts_func(examples):
    inputs = examples["Sentence"]
    outputs = examples["Response"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = prompt_template.format(input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

try:
    logger.info("Creating dataset...")
    dataset = Dataset.from_pandas(df)
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(eval_dataset)}")
    logger.info(f"Sample train data: {train_dataset[0]}")
    
    # Log tokenized sample
    sample_prompt = train_dataset[0]["text"]
    tokenized = tokenizer(sample_prompt, return_tensors="pt")
    logger.info(f"Sample tokenized input IDs: {tokenized.input_ids[0].tolist()[:50]}...")
    logger.info(f"Sample decoded tokens: {tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)}")
except Exception as e:
    logger.error(f"Error creating dataset: {str(e)}")
    raise

# -------------------------------
# Step 6: Apply LoRA
# -------------------------------
try:
    logger.info("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
except Exception as e:
    logger.error(f"Error applying LoRA: {str(e)}")
    raise

# -------------------------------
# Step 7: TrainingArguments
# -----------------------
try:
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )
except Exception as e:
    logger.error(f"Error setting up training arguments: {str(e)}")
    raise

# -------------------------------
# Step 8: Custom callback for logging
# -------------------------------
class DataLoggingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Training started")
        logger.info("Skipping first batch logging due to callback limitations")

# -------------------------------
# Step 9: Train the model
# -------------------------------
try:
    logger.info("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        callbacks=[DataLoggingCallback()],
    )
    
    # Log sample prediction
    def log_sample_prediction(stage):
        sample_input = train_dataset[0]["Sentence"]
        prompt = prompt_template.format(sample_input, "") + tokenizer.eos_token
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("### Output:")[-1].strip() if "### Output:" in decoded else decoded.strip()
        logger.info(f"{stage} prediction: Input: {sample_input}\nOutput: {response}")
    
    logger.info("Pre-training sample prediction...")
    log_sample_prediction("Pre-training")
    
    trainer.train()
    
    logger.info("Post-training sample prediction...")
    log_sample_prediction("Post-training")
    
    # Log first training batch
    logger.info("Logging first training batch post-training...")
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"][0].tolist()[:50]
    logger.info(f"Post-training batch input IDs: {input_ids}...")
    logger.info(f"Post-training batch decoded: {tokenizer.decode(input_ids, skip_special_tokens=True)}")
except Exception as e:
    logger.error(f"Error during training: {str(e)}")
    raise

# -------------------------------
# Step 10: Save model
# -------------------------------
try:
    logger.info("Saving model...")
    model.save_pretrained("smart-home-lora-llama")
    tokenizer.save_pretrained("smart-home-lora-llama")
except Exception as e:
    logger.error(f"Error saving model: {str(e)}")
    raise

# -------------------------------
# Step 11: Test the model
# -------------------------------
try:
    logger.info("Testing model...")
    def generate_response(question, max_new_tokens=50):
        prompt = prompt_template.format(question, "")  # <-- Remove eos_token here!
        logger.info(f"Test prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(inputs.input_ids.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        logger.info(f"Raw output tokens: {outputs[0].tolist()}")
        logger.info(f"Token-to-text: {tokenizer.convert_ids_to_tokens(outputs[0].tolist())}")
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Decoded output: {decoded}")
        response = decoded.split("### Output:")[-1].strip() if "### Output:" in decoded else decoded.strip()
        return response

    # Test 1: Lighting query
    test_question = "Turn on the music"
    response = generate_response(test_question)
    logger.info(f"💡 Test Question 1: {test_question}\n Model Response 1: {response}")

    # Test 2: Dataset lighting example
    test_question_2 = train_dataset[0]["Sentence"]
    response_2 = generate_response(test_question_2)
    logger.info(f"💡 Test Question 2: {test_question_2}\n Model Response 2: {response_2}")
except Exception as e:
    logger.error(f"Error testing model: {str(e)}")
    raise