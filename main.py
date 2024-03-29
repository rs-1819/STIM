import torch
import torch.optim as optim
import transformers
from torch.utils.data import DataLoader
import torch.nn.functional as F

def train_softprompt(model, tokenizer, train_data, batch_size, num_epochs, device):
    # Create a DataLoader for training data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize softprompt embeddings
    prompt_size = 5
    softprompt_embeddings = torch.randn(prompt_size, model.config.d_model, device=device, requires_grad=True)
    input_ids = torch.arange(0, prompt_size, dtype=torch.long, device=device).unsqueeze(0)

    # Set optimizer
    # Set a lower learning rate
    optimizer = optim.Adam([softprompt_embeddings], lr=5e-5)  # Adjust as needed

# Inside your training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Tokenize batch and convert it into a tensor
            batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

            # Prepare input text by concatenating it with the softprompt tokens
            inputs_with_prompt = torch.cat((input_ids.expand(batch.size(0), -1), batch), dim=1)

            # Create labels by shifting input_ids to the right
            labels = batch.roll(shifts=-1, dims=-1)
            labels[:, -1] = -100  # Ignore loss for the last token

            # Forward pass to get logits
            outputs = model(inputs_with_prompt).logits

            # Ensure outputs and labels are aligned
            output_length = outputs.size(1)
            label_length = labels.size(1)

            if output_length != label_length:
                min_length = min(output_length, label_length)
                outputs = outputs[:, :min_length]
                labels = labels[:, :min_length]

            # Compute loss manually using Cross-Entropy
            loss = F.cross_entropy(outputs.transpose(1, 2), labels, ignore_index=-100)

            if loss is not None:
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            else:
                print("Loss is None. Skipping backward pass and optimizer step.")

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] Training Loss: {avg_loss}")


    # Return the trained softprompt embeddings
    return softprompt_embeddings

# Load model, tokenizer, and configure device
name = 'mosaicml/mpt-1b-redpajama-200b'
config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(name, config=config, torch_dtype=torch.float32, trust_remote_code=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define training data and train the softprompt
train_data = [
    "Once upon a time in a land far away, there was a king and a queen.",
    "Long ago, in a small village, a young boy found a magical sword.",
    "In the depths of the forest, a strange creature guarded a hidden treasure.",
    "A brave knight set out on a quest to rescue a princess from a dragon.",
    "In a distant kingdom, a sorcerer plotted to overthrow the throne.",
    "A curious cat in a bustling market town discovered a mysterious amulet.",
    "Under the light of the full moon, a werewolf roamed the countryside.",
    "A wandering minstrel sang tales of ancient heroes and lost cities.",
    "In the heart of the mountains, a hidden cave held an ancient secret.",
    "A young girl with magical powers befriended a group of talking animals.",
    "A powerful wizard cast a spell to protect his land from dark forces.",
    "A pirate captain sailed the seven seas in search of a fabled island.",
    "An enchanted forest was home to fairies, sprites, and mystical creatures.",
    "A fierce battle ensued between rival clans over an ancestral land.",
    "A mysterious traveler arrived in town, bringing tales of distant worlds.",
    "At the stroke of midnight, a curse transformed a prince into a beast.",
    "A group of adventurers found a map leading to a sunken treasure.",
    "In the ruins of an old castle, a ghostly apparition appeared at night.",
    "A mermaid sought the help of a sea witch to visit the land above.",
    "A prophecy foretold the arrival of a hero who would save the kingdom.",
    "A band of thieves plotted to steal the crown jewels during the festival.",
    "In a land of perpetual winter, a queen with ice powers lived in solitude.",
    "A dragon hoarded gold and jewels in a cavern deep beneath a mountain.",
    "A witch in a gingerbread house lured unsuspecting travelers.",
    "Two star-crossed lovers met secretly in the palace gardens.",
    "A magic mirror revealed the truth hidden behind a web of lies.",
    "An ancient tree in the village square whispered secrets of the past.",
    "A loyal squire embarked on a journey to prove his knighthood.",
    "A crafty goblin traded riddles with those who crossed his bridge.",
    "A forgotten tomb held the key to an age-old mystery.",
    "A fierce storm at sea revealed a path to a hidden underwater city.",
    "A queen disguised herself as a commoner to understand her people.",
    "In a moonlit glade, a fairy granted wishes to the pure of heart.",
    "A legendary phoenix was reborn from its ashes in a display of light."]
softprompt_embeddings = train_softprompt(model, tokenizer, train_data, batch_size=1, num_epochs=10, device=device)

# Save the softprompt embeddings
path = "./softprompt_embeddings.pth"
torch.save(softprompt_embeddings, path)
