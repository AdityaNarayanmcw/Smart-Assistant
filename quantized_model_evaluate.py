#converting to 8bit quantized model from finedtune_llama3.2.1-merged


# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# from typing import Dict

# def get_directory_size(path: str) -> float:
#     """Calculate the size of a directory in GiB."""
#     total = 0
#     for dirpath, _, filenames in os.walk(path):
#         for f in filenames:
#             fp = os.path.join(dirpath, f)
#             total += os.path.getsize(fp)
#     return total / 1e9  # Convert bytes to GiB

# def quantize_model(
#     base_model_name: str,
#     lora_model_path: str,
#     merged_model_path: str,
#     quantized_model_path: str
# ) -> Dict[str, float]:
#     """Merge LoRA adapters with base model, save merged model, and quantize it."""
#     try:
#         # Load tokenizer
#         print(f"Loading tokenizer from {base_model_name}...")
#         tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.padding_side = "right"
#         tokenizer.truncation_side = "right"
        
#         # Load base model
#         print(f"Loading base model {base_model_name}...")
#         base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_name,
#             torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True,
#             device_map="auto"
#         )
        
#         # Load LoRA adapters
#         print(f"Loading LoRA adapters from {lora_model_path}...")
#         model = PeftModel.from_pretrained(base_model, lora_model_path)
        
#         # Merge LoRA adapters with base model
#         print(f"Merging LoRA adapters...")
#         merged_model = model.merge_and_unload()
        
#         # Measure original (merged) model size
#         os.makedirs(merged_model_path, exist_ok=True)
#         print(f"Saving merged model to {merged_model_path}...")
#         merged_model.save_pretrained(merged_model_path)
#         tokenizer.save_pretrained(merged_model_path)
#         original_size = get_directory_size(merged_model_path)
        
#         # Configure 8-bit quantization
#         quantization_config = BitsAndBytesConfig(
#             load_in_8bit=True,
#             llm_int8_threshold=6.0
#         )
        
#         # Load merged model with quantization
#         print(f"Loading merged model with 8-bit quantization...")
#         quantized_model = AutoModelForCausalLM.from_pretrained(
#             merged_model_path,
#             quantization_config=quantization_config,
#             device_map="auto",
#             low_cpu_mem_usage=True
#         )
        
#         # Save quantized model
#         os.makedirs(quantized_model_path, exist_ok=True)
#         print(f"Saving quantized model to {quantized_model_path}...")
#         quantized_model.save_pretrained(quantized_model_path)
#         tokenizer.save_pretrained(quantized_model_path)
        
#         # Measure quantized model size
#         quantized_size = get_directory_size(quantized_model_path)
        
#         # Clean up
#         del base_model
#         del model
#         del merged_model
#         del quantized_model
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
        
#         return {
#             "original_size_gib": original_size,
#             "quantized_size_gib": quantized_size,
#             "size_reduction_percent": ((original_size - quantized_size) / original_size * 100) if original_size > 0 else 0
#         }
#     except Exception as e:
#         print(f"Error during quantization: {e}")
#         return {}

# if __name__ == "__main__":
#     base_model_name = "meta-llama/Llama-3.2-1B"  # Replace with your base model path or Hugging Face ID
#     lora_model_path = "./finetuned_llama321/final_model"
#     merged_model_path = "./finetuned_llama321_merged"
#     quantized_model_path = "./finetuned_llama321_8bit"
    
#     results = quantize_model(
#         base_model_name,
#         lora_model_path,
#         merged_model_path,
#         quantized_model_path
#     )
#     if results:
#         print("\nQuantization Results:")
#         print(f"Original (Merged) Model Size: {results['original_size_gib']:.2f} GiB")
#         print(f"Quantized Model Size: {results['quantized_size_gib']:.2f} GiB")
#         print(f"Size Reduction: {results['size_reduction_percent']:.2f}%")
#     else:
#         print("Quantization failed.")


#converting to 4bit quantized model from finedtune_llama3.2.1-merged
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Dict

def get_directory_size(path: str) -> float:
    """Calculate the size of a directory in GiB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / 1e9  # Convert bytes to GiB

def quantize_model(
    base_model_name: str,
    lora_model_path: str,
    merged_model_path: str,
    quantized_model_path: str
) -> Dict[str, float]:
    """Merge LoRA adapters with base model, save merged model, and quantize it to 4-bit."""
    try:
        # Load tokenizer
        print(f"Loading tokenizer from {base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        
        # Load base model
        print(f"Loading base model {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Load LoRA adapters
        print(f"Loading LoRA adapters from {lora_model_path}...")
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        
        # Merge LoRA adapters with base model
        print(f"Merging LoRA adapters...")
        merged_model = model.merge_and_unload()
        
        # Measure original (merged) model size
        os.makedirs(merged_model_path, exist_ok=True)
        print(f"Saving merged model to {merged_model_path}...")
        merged_model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)
        original_size = get_directory_size(merged_model_path)
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Use NF4 quantization
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True  # Enable double quantization for better accuracy
        )
        
        # Load merged model with quantization
        print(f"Loading merged model with 4-bit quantization...")
        quantized_model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Save quantized model
        os.makedirs(quantized_model_path, exist_ok=True)
        print(f"Saving quantized model to {quantized_model_path}...")
        quantized_model.save_pretrained(quantized_model_path)
        tokenizer.save_pretrained(quantized_model_path)
        
        # Measure quantized model size
        quantized_size = get_directory_size(quantized_model_path)
        
        # Clean up
        del base_model
        del model
        del merged_model
        del quantized_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "original_size_gib": original_size,
            "quantized_size_gib": quantized_size,
            "size_reduction_percent": ((original_size - quantized_size) / original_size * 100) if original_size > 0 else 0
        }
    except Exception as e:
        print(f"Error during quantization: {e}")
        return {}

if __name__ == "__main__":
    base_model_name = "meta-llama/Llama-3.2-1B"  # Replace with your base model path or Hugging Face ID
    lora_model_path = "./finetuned_llama321/final_model"
    merged_model_path = "./finetuned_llama321_merged"
    quantized_model_path = "./finetuned_llama321_4bit"
    
    results = quantize_model(
        base_model_name,
        lora_model_path,
        merged_model_path,
        quantized_model_path
    )
    if results:
        print("\nQuantization Results:")
        print(f"Original (Merged) Model Size: {results['original_size_gib']:.2f} GiB")
        print(f"Quantized Model Size: {results['quantized_size_gib']:.2f} GiB")
        print(f"Size Reduction: {results['size_reduction_percent']:.2f}%")
    else:
        print("Quantization failed.")