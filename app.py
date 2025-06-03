import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Function to check GPU memory usage
def print_gpu_memory(step_name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"Memory at {step_name}: Allocated {allocated:.2f} GiB, Reserved {reserved:.2f} GiB, Free {total - allocated:.2f} GiB"

# Function to clean response (remove timestamps and special tokens)
def clean_response(response):
    # Remove timestamps (e.g., "2023-02-20 14:45:12")
    response = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.?\d*', '', response)
    # Remove special tokens (e.g., "<|END|>")
    response = re.sub(r'<\|\w+\|>', '', response)
    # Remove redundant phrases and extra whitespace
    response = re.sub(r'\s+', ' ', response).strip()
    return response

# Function to infer Category, Subcategory, and Action
def infer_fields(sentence):
    sentence = sentence.lower()
    
    # Define keyword mappings with stricter matching
    category_map = {
        r'\b(light|bright|dim|illuminate)\b': 'lights',
        r'\b(temperature|heat|cool|thermostat)\b': 'thermostat',
        r'\b(fan|airflow)\b': 'fan',
        r'\b(lock|unlock|security)\b': 'security'
    }
    
    subcategory_map = {
        r'\b(basement)\b': 'basement',
        r'\b(kitchen)\b': 'kitchen',
        r'\b(living\s*room)\b': 'living_room',
        r'\b(bedroom)\b': 'bedroom',
        r'\b(bathroom)\b': 'bathroom',
        r'\b(hallway)\b': 'hallway'
    }
    
    action_map = {
        r'\b(bright|more\s*bright|turn\s*on|illuminate)\b': 'on',
        r'\b(dim|less\s*bright|turn\s*off)\b': 'off',
        r'\b(lock)\b': 'lock',
        r'\b(unlock)\b': 'unlock',
        r'\b(set|adjust)\b': 'set'
    }
    
    # Infer Category
    category = 'general'
    for pattern, value in category_map.items():
        if re.search(pattern, sentence):
            category = value
            break
    
    # Infer Subcategory
    subcategory = 'unknown'
    for pattern, value in subcategory_map.items():
        if re.search(pattern, sentence):
            subcategory = value
            break
    
    # Infer Action
    action = 'execute'
    for pattern, value in action_map.items():
        if re.search(pattern, sentence):
            action = value
            break
    
    logger.info(f"Inferred fields - Sentence: {sentence}, Category: {category}, Subcategory: {subcategory}, Action: {action}")
    return category, subcategory, action

# Cache model and tokenizer to load only once
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "./finetuned_llama"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        st.success("Model and tokenizer loaded successfully.")
        st.write(print_gpu_memory("Model Load"))
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Model loading failed: {str(e)}")
        return None, None

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Streamlit app
st.title("Smart Assistant for Home Automation")
st.write("Enter a command below (e.g., 'Can you make it more bright in the basement in 5 minutes?') to get a response from the fine-tuned Llama-3.2-3B model.")

# Input form
with st.form(key="input_form"):
    sentence = st.text_area("Command", value="", placeholder="e.g., Can you make it more bright in the basement in 5 minutes?")
    submit_button = st.form_submit_button(label="Generate Response")

# Process input and generate response
if submit_button:
    if model is None or tokenizer is None:
        st.error("Model or tokenizer not loaded. Please check the server configuration.")
    else:
        if not sentence:
            st.error("Please enter a command.")
        else:
            with st.spinner("Generating response..."):
                try:
                    # Infer fields
                    category, subcategory, action = infer_fields(sentence)
                    
                    # Create input prompt
                    input_text = (f"Category: {category}, Subcategory: {subcategory}, "
                                 f"Action: {action}, Sentence: {sentence}")
                    prompt = (f"You are a smart assistant. Respond concisely, preserving temporal details "
                             f"in the user's instruction. User: {input_text} ### Assistant: ")
                    
                    # Tokenize and generate
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,  # Reduced to enforce conciseness
                        num_beams=5,
                        no_repeat_ngram_size=3,
                        do_sample=False,
                        temperature=0.6  # Lowered for more deterministic output
                    )
                    generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Assistant:")[-1].strip()
                    generated_response = clean_response(generated_response.split("###")[0].strip())
                    
                    # Display inferred fields and response
                    st.subheader("Inferred Fields")
                    st.write(f"Category: {category}")
                    st.write(f"Subcategory: {subcategory}")
                    st.write(f"Action: {action}")
                    st.subheader("Generated Response")
                    st.write(generated_response)
                    
                    # Display memory usage
                    st.write(print_gpu_memory("Response Generation"))
                    
                    # Log response
                    logger.info(f"Generated response: {generated_response}")
                    
                    # Clear memory
                    del inputs, outputs
                    clear_gpu_memory()
                    
                except torch.cuda.OutOfMemoryError as e:
                    st.error(f"CUDA OOM Error: {str(e)}. Clearing GPU memory and retrying may help.")
                    logger.error(f"CUDA OOM Error: {str(e)}")
                    clear_gpu_memory()
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    logger.error(f"Generation error: {str(e)}")

# Clear memory on app close
if st.button("Clear GPU Memory"):
    clear_gpu_memory()
    st.success("GPU memory cleared.")