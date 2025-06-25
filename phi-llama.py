import os
import logging
import pandas as pd
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

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
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    logger.error(f"Error checking environment: {str(e)}")
    raise

# -------------------------------
# Step 4: Load model & tokenizer
# -------------------------------
max_seq_length = 1024
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
    df = pd.read_csv("/home/mcw/Aditya/unsloth/light_sent_Resp.csv")
    required_columns = ['Sentence', 'Response']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV must contain 'Sentence' and 'Response' columns")
    logger.info(f"Loaded {len(df)} examples from light.csv")
    logger.info(f"Sample row: {df.iloc[0].to_dict()}")
except Exception as e:
    logger.error(f"Error loading CSV: {str(e)}")
    raise

# Simple prompt format
prompt_template = """Question: {}
Response: {}"""

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
    dataset = dataset.map(formatting_prompts_func, batched=True)
    logger.info(f"Sample formatted data: {dataset[0]}")
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
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
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
        per_device_train_batch_size=2,  # Matches your log
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=3,  # Increased for better learning
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        gradient_checkpointing=True,
    )
except Exception as e:
    logger.error(f"Error setting up training arguments: {str(e)}")
    raise

# -------------------------------
# Step 8: Train the model
# -------------------------------
try:
    logger.info("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    trainer.train()
except Exception as e:
    logger.error(f"Error during training: {str(e)}")
    raise

# -------------------------------
# Step 9: Save model
# -------------------------------
try:
    logger.info("Saving model...")
    model.save_pretrained("smart-home-lora-llama")
    tokenizer.save_pretrained("smart-home-lora-llama")
except Exception as e:
    logger.error(f"Error saving model: {str(e)}")
    raise

# -------------------------------
# Step 10: Test the model
# -------------------------------
try:
    logger.info("Testing model...")
    def generate_response(question, max_new_tokens=300):
        prompt = prompt_template.format(question, "") + tokenizer.eos_token
        logger.info(f"Test prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(inputs.input_ids.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            min_length=10,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
        )
        logger.info(f"Raw output tokens: {outputs[0].tolist()}")
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Decoded output: {decoded}")
        response = decoded.split("Response:")[-1].strip() if "Response:" in decoded else decoded.strip()
        return response

    # Test 1: Lighting query
    test_question = "Turn on the kitchen lights"
    response = generate_response(test_question)
    logger.info(f"💡 Test Question 1: {test_question}\n🔁 Model Response 1: {response}")

    # Test 2: Dataset lighting example
    test_question_2 = df["Sentence"].iloc[0]
    response_2 = generate_response(test_question_2)
    logger.info(f"💡 Test Question 2: {test_question_2}\n🔁 Model Response 2: {response_2}")

    # Test 3: Camera query
    test_question_3 = "Was there movement on the camera in the kitchen yesterday?"
    response_3 = generate_response(test_question_3)
    logger.info(f"💡 Test Question 3: {test_question_3}\n🔁 Model Response 3: {response_3}")
except Exception as e:
    logger.error(f"Error testing model: {str(e)}")
    raise