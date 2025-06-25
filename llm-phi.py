import pandas as pd

# Define the column headers based on the provided data
columns = [
    "Model",
    "ROUGE-L (%) Mean", "ROUGE-L (%) Std",
    "Cosine Similarity (%) Mean", "Cosine Similarity (%) Std",
    "Edit Distance Similarity (%) Mean", "Edit Distance Similarity (%) Std",
    "Latency (sec) Mean", "Latency (sec) Std",
    "Generation Time (sec) Mean", "Generation Time (sec) Std",
    "Throughput (tokens/sec) Mean", "Throughput (tokens/sec) Std",
    "Memory Before (GiB) Mean", "Memory Before (GiB) Max",
    "Memory After (GiB) Mean", "Memory After (GiB) Max"
]

# Define the data rows
data = [
    [
        "Llama-3.2-1B", 88.37, 20.78, 99.19, 4.14, 83.93, 21.11, 0.0, 0.0,
        0.54, 0.21, 35.53, 5.91, 2.48, 2.48, 2.48, 2.48
    ],
    [
        "Llama-3.2-3B", 87.59, 21.08, 99.14, 4.26, 83.98, 20.83, 0.0, 0.0,
        1.2, 0.41, 15.78, 2.33, 6.44, 6.44, 6.44, 6.44
    ]
]

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to Excel
output_file = "model_comparison_summary.xlsx"
try:
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Excel file saved successfully as {output_file}")
except Exception as e:
    print(f"Error saving Excel file: {e}")

# Display the DataFrame for verification
print("\nDataFrame Preview:")
print(df.to_string())