import torch
import transformers

def load_softprompt_embedding(embedding_file, tokenizer):
    softprompt_embedding = torch.load(embedding_file)
    prompt_tokens = tokenizer(embedding_file.split(".")[0], return_tensors="pt")
    input_ids = prompt_tokens["input_ids"]
    return input_ids, softprompt_embedding

# Model name
name = 'mosaicml/mpt-7b-storywriter'

# Softprompt embedding pt file
softprompt_embedding_file = "path/to/softprompt_embedding.pt"

# Load the model
config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda:0'
model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=True)

# Load softprompt embedding
input_ids, softprompt_embedding = load_softprompt_embedding(softprompt_embedding_file, tokenizer)

# Insert softprompt embedding into model
model.transformer.wte.embedding.weight.data[input_ids] = softprompt_embedding.to(model.transformer.wte.embedding.weight.device)

# Inference example
sample_sentence = "Once upon a time"
input_tokens = tokenizer.encode(sample_sentence, return_tensors="pt")
inputs_with_prompt = torch.cat((input_ids, input_tokens), dim=1)
outputs = model.generate(inputs_with_prompt, max_length=50)

decoded_output = tokenizer.decode(outputs[0])
print(decoded_output)