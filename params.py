
# Hugging Face Details
HUGGINGFACE_MODEL = "microsoft/phi-2"

# Choose Your Provider: 'huggingface' or 'custom'
PROVIDER = "huggingface"

# Generation Details (for data output)
OUTPUT_FILE_PATH = "dataset.jsonl"
OUTPUT_CSV_PATH = "dataset.csv"

#The current setup uses NUM_WORKERS threads for parallel data generation. 
NUM_WORKERS = 9