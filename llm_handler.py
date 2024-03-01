
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from params import HUGGINGFACE_MODEL, PROVIDER

# Set the default device based on availability
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Initialize Hugging Face model and tokenizer
model = AutoModelForCausalLM.from_pretrained(HUGGINGFACE_MODEL, torch_dtype="auto", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL, trust_remote_code=True)

def send_to_huggingface(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt", return_attention_mask=False).to(device)
    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    return text

def send_to_llm(provider, msg_list):
    if provider == "huggingface":
        # Assuming msg_list is a list of messages and we take the last user message as the prompt
        prompt_text = msg_list[-1]['content'] if msg_list else ''
        response = send_to_huggingface(prompt_text)
        return response, None  # No usage information for Hugging Face in this context
    else:
        raise ValueError("Unsupported provider: {}".format(provider))

