import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import math
import copy
from tqdm import tqdm
import os
import json
import random
from torchmetrics.text import BLEUScore
from sklearn.model_selection import train_test_split # New import for splitting data

# --- Configuration ---
# Define paths relative to the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "merged_f.txt") 
CHECKPOINT_DIR = os.path.join(BASE_DIR, "nmt_checkpoints")
TOKENIZER_FILE = os.path.join(BASE_DIR, "sinhala_tokenizer.json")

# Ensure the data file exists or provide a placeholder. 
# You MUST place your merged_f.txt file in the same directory as this script.
if not os.path.exists(DATA_FILE):
    print(f"ERROR: Data file not found at {DATA_FILE}")
    print("Please ensure 'merged_f.txt' is in the same folder as this script.")
    exit()

# Model Hyperparameters
src_vocab_size = 10000 # Placeholder: Will be updated by tokenizer.next_id for src
tgt_vocab_size = 2000  # Placeholder: This is likely fixed based on your target ID system (assuming target IDs are pre-tokenized)
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1
num_epochs = 10
BATCH_SIZE = 16

# --- Transformer Components ---

class MultiHeadAttention(nn.Module):
  """Multi-Head Attention mechanism."""

  def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

  def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # -1e9 is used for numerical stability when applying softmax to large negative numbers
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

  def split_heads(self, x):
    """Reshapes the input for parallel computation across attention heads."""
    batch_size, seq_length, d_model = x.size()
    # (batch_size, seq_length, num_heads, d_k) -> (batch_size, num_heads, seq_length, d_k)
    return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

  def combine_heads(self, x):
    """Combines results from multiple heads back into a single tensor."""
    batch_size, _, seq_length, d_k = x.size()
    # (batch_size, num_heads, seq_length, d_k) -> (batch_size, seq_length, num_heads, d_k) -> (batch_size, seq_length, d_model)
    return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

  def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    """A two-layer feed-forward network applied to each position separately."""
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    """Adds sinusoidal position information to the token embeddings."""
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Calculates a scale factor for the sine/cosine arguments
        # div_term = 1 / (10000 ^ (2i / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term) # Apply cosine to odd indices

        # pe.unsqueeze(0) makes it (1, max_seq_length, d_model) for broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    """A single layer of the Transformer Encoder."""
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model) # Applied after first sub-layer (Attention + Dropout)
        self.norm2 = nn.LayerNorm(d_model) # Applied after second sub-layer (FFN + Dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. Self-Attention Sub-layer with Residual Connection & Normalization
        attn_output = self.self_attn(x, x, x, mask)
        # x_attn = LayerNorm(x + Dropout(SelfAttention(x)))
        x = self.norm1(x + self.dropout(attn_output)) 

        # 2. Feed-Forward Sub-layer with Residual Connection & Normalization
        ff_output = self.feed_forward(x)
        # x_ffn = LayerNorm(x_attn + Dropout(FeedForward(x_attn)))
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    """A single layer of the Transformer Decoder."""
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads) # Attends to encoder output
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model) # After self-attention
        self.norm2 = nn.LayerNorm(d_model) # After cross-attention
        self.norm3 = nn.LayerNorm(d_model) # After feed-forward
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 1. Masked Self-Attention Sub-layer
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Cross-Attention Sub-layer (Query from decoder, Key/Value from encoder)
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 3. Feed-Forward Sub-layer
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    """The complete Transformer model (Encoder-Decoder)."""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Creates a stack of N=num_layers EncoderLayer/DecoderLayer modules
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size) # Output projection
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """Creates masks: Source padding mask and Target combined padding/look-ahead mask."""
        # Source mask (for encoder self-attention and decoder cross-attention)
        # (batch_size, 1, 1, src_seq_len) - mask=0 where src==<pad>
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) 

        # Target padding mask (for decoder self-attention)
        # (batch_size, 1, tgt_seq_len, 1) - mask=0 where tgt==<pad>
        tgt_mask_padding = (tgt != 0).unsqueeze(1).unsqueeze(3) 
        
        # Look-ahead mask (upper triangular part is 1, lower is 0, diagonal is 0)
        # This prevents the decoder from attending to future tokens.
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
        
        # Combined target mask
        tgt_mask = tgt_mask_padding & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encoder Input: Embedding + Positional Encoding + Dropout
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        
        # Decoder Input: Embedding + Positional Encoding + Dropout
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Run through Encoder stack
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Run through Decoder stack
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            # Cross-attention attends to encoder output (enc_output, enc_output)
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # Final linear layer to project to target vocabulary size
        output = self.fc(dec_output)
        return output

# --- Data Handling Classes ---

def load_custom_dataset(path):
    """Loads source sentences and their pre-tokenized target IDs from the data file."""
    src_list = []
    tgt_list = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Check for the expected separator, as in the original notebook
            if "@" not in line:
                continue

            try:
                src, tgt = line.strip().split("@")
                tgt_tokens = tgt.split("|")
                
                ids = []
                for tok in tgt_tokens:
                    # Target IDs are the part before the first ":"
                    parts = tok.split(":")
                    if parts and parts[0].isdigit():
                        ids.append(int(parts[0]))
                    else:
                        ids.append(3)   # <unk> = 3

                src_list.append(src.strip())
                tgt_list.append(ids)
            except Exception as e:
                # print(f"Skipping malformed line: {line.strip()} due to {e}")
                continue

    return src_list, tgt_list

class SinhalaTokenizer:
    """A simple vocabulary builder and encoder for the Sinhala source text."""
    def __init__(self):
        # Default special tokens
        self.w2i = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3} 
        self.i2w = {0:"<pad>",1:"<bos>",2:"<eos>",3:"<unk>"}
        self.next_id = 4 # Start regular tokens from 4

    def build(self, sentences):
        """Populates the vocabulary based on a list of source sentences."""
        for s in sentences:
            for w in s.split():
                if w not in self.w2i:
                    self.w2i[w] = self.next_id
                    self.i2w[self.next_id] = w
                    self.next_id += 1

    def encode(self, text):
        """Converts a source sentence (string) into a sequence of token IDs."""
        ids = [1]  # <bos>
        for w in text.split():
            ids.append(self.w2i.get(w, 3)) # Use 3 (<unk>) if word is not found
        ids.append(2)  # <eos>
        return ids
    
    def ids_to_text(self, id_list, filter_special=False):
        """Converts a sequence of token IDs back to a string."""
        words = []
        for idx in id_list:
            word = self.i2w.get(idx, "<unk>")
            if filter_special and idx in [0, 1, 2]: # <pad>, <bos>, <eos>
                 continue
            if idx == 3 and filter_special: # <unk>
                continue
            words.append(word)
        return " ".join(words)

class MyDataset(data.Dataset):
    """PyTorch Dataset for the NMT task."""
    def __init__(self, src_texts, tgt_ids, tokenizer):
        self.src = src_texts
        self.tgt = tgt_ids
        self.tok = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        # Encode source text using the tokenizer (adds <bos>, <eos>)
        s = self.tok.encode(self.src[idx]) 
        # Target IDs are pre-tokenized, just add <bos> and <eos>
        t = [1] + self.tgt[idx] + [2] 
        return torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long)

def collate_fn(batch):
    """Pads sequences within a batch to the longest sequence length."""
    # srcs: List of (s_tensor, t_tensor)
    srcs, tgts = zip(*batch) 

    max_s = max(len(s) for s in srcs)
    max_t = max(len(t) for t in tgts)

    padded_s = []
    padded_t = []

    # Padding is done with 0 (<pad> token ID)
    for s, t in zip(srcs, tgts):
        # torch.zeros(N, dtype=torch.long) creates a padding tensor
        ps = torch.cat([s, torch.zeros(max_s - len(s), dtype=torch.long)])
        pt = torch.cat([t, torch.zeros(max_t - len(t), dtype=torch.long)])
        padded_s.append(ps)
        padded_t.append(pt)

    # torch.stack creates a tensor of shape (batch_size, max_seq_len)
    return torch.stack(padded_s), torch.stack(padded_t)

# --- Decoding and Evaluation ---

def greedy_decode(model, src, tokenizer, max_len=50, device='cpu'):
    """Performs inference using greedy decoding."""
    model.eval()

    # 1. Tokenize and prepare source tensor
    src_ids = tokenizer.encode(src)
    # Unsqueeze(0) for batch dimension: (1, seq_len)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device) 

    # 2. Compute Encoder Output
    with torch.no_grad():
        # Source Mask for encoder
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2).to(device)
        
        # Encoding process
        src_embedded = model.dropout(model.positional_encoding(model.encoder_embedding(src_tensor)))
        enc_output = src_embedded
        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

    # 3. Initialize Decoder Input with <bos> (index 1)
    # (1, 1) -> batch_size=1, seq_len=1
    decoder_input = torch.tensor([[1]], dtype=torch.long).to(device) 

    generated_tokens = []

    # 4. Loop to generate tokens
    for _ in range(max_len):
        # Create Target Mask (padding + look-ahead)
        tgt_mask_padding = (decoder_input != 0).unsqueeze(1).unsqueeze(3)
        seq_length = decoder_input.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask_padding & nopeak_mask

        # Run Decoder
        with torch.no_grad():
            tgt_embedded = model.dropout(model.positional_encoding(model.decoder_embedding(decoder_input)))
            dec_output = tgt_embedded
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

            # Project to vocabulary: (1, seq_len, vocab_size)
            output = model.fc(dec_output) 

        # Get the token with the highest probability for the LAST position
        next_token_logits = output[:, -1, :] # Logits for the last generated token
        next_token_id = next_token_logits.argmax(dim=-1).item()

        # Stop if <eos> (index 2) is generated
        if next_token_id == 2:
            break

        # Append the new token to the sequence and update the decoder input
        generated_tokens.append(next_token_id)
        # Concatenate: decoder_input (1, N) + new_token (1, 1) -> (1, N+1)
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)

    return generated_tokens

def evaluate_model(model, src_sentences, tgt_ids, tokenizer, device='cpu'):
    """Calculates BLEU score on a test set."""
    model.eval()
    # Initialize BLEU score metric (can be set to different weights if needed)
    bleu = BLEUScore(n_gram=4, smooth=False) 

    predicted_sentences = []
    actual_sentences = []
    
    print("\n-------------------------------\nSTARTING EVALUATION\n-------------------------------")
    print("Generating predictions...")
    
    # Use only a subset for quick check if the full dataset is too large/slow
    num_samples_to_test = min(100, len(src_sentences)) 
    
    for i in tqdm(range(num_samples_to_test)):
        src_text = src_sentences[i]
        target_id_list = tgt_ids[i]

        # 1. Get Prediction (Greedy Decode)
        pred_ids = greedy_decode(model, src_text, tokenizer, max_len=50, device=device)

        # 2. Convert IDs to Strings (BLEU metric from torchmetrics expects lists of tokens or strings)
        # We convert the list of IDs to a string representation for comparison.
        # This keeps the comparison purely on the target ID sequence.
        pred_str = " ".join([str(x) for x in pred_ids if x not in [0,1,2]])
        target_str = " ".join([str(x) for x in target_id_list])

        predicted_sentences.append(pred_str)
        # BLEU expects a list of reference translations for each hypothesis
        actual_sentences.append([target_str]) 

    # 3. Calculate Score
    score = bleu(predicted_sentences, actual_sentences)
    print(f"\n-------------------------------\nFinal BLEU Score (on {num_samples_to_test} samples): {score.item():.4f}\n-------------------------------")

    # Show a few examples
    for i in range(min(5, num_samples_to_test)):
        print(f"Src: {src_sentences[i]}")
        # Note: If you want to convert IDs to text, you need to map them outside of this script.
        print(f"Ref IDs: {actual_sentences[i][0]}")
        print(f"Pred IDs: {predicted_sentences[i]}")
        print("-" * 10)


# --- Main Execution ---

def main():
    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Create required directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 3. Load and Split Data
    print(f"Loading data from {DATA_FILE}...")
    src_sentences_all, tgt_ids_all = load_custom_dataset(DATA_FILE)
    print(f"Total samples loaded: {len(src_sentences_all)}")

    if len(src_sentences_all) == 0:
        print("No valid data found. Exiting.")
        return

    # Split data into training and testing sets
    # Using a 90/10 split (can be adjusted)
    # train_size = 0.9, test_size = 0.1
    train_src, test_src, train_tgt, test_tgt = train_test_split(
        src_sentences_all, tgt_ids_all, test_size=0.1, random_state=42
    )
    print(f"Training samples: {len(train_src)}")
    print(f"Testing samples: {len(test_src)}")

    # 4. Tokenizer Setup
    tokenizer = SinhalaTokenizer()
    tokenizer.build(train_src) # Build vocab only on training source data

    # Update src_vocab_size based on the tokenizer
    global src_vocab_size
    src_vocab_size = tokenizer.next_id
    print(f"Source Vocabulary Size: {src_vocab_size}")
    # Target vocabulary size remains the original placeholder as it's handled by your pre-tokenization

    # 5. Dataset and DataLoader Setup
    train_dataset = MyDataset(train_src, train_tgt, tokenizer)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # 6. Model Initialization
    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout
    ).to(device)

    # 7. Training Setup
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore_index=0 ignores <pad> token
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    start_epoch = 0
    global_step = 0
    last_batch_idx = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
    
    # 8. Load Checkpoint (Resuming Training)
    if os.path.exists(latest_ckpt):
        print("🔄 Loading previous checkpoint...")
        # Load map_location='cpu' first, then move model to device
        checkpoint = torch.load(latest_ckpt, map_location=device) 
        
        # Load state dicts
        try:
            transformer.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"]
            global_step = checkpoint["global_step"]
            last_batch_idx = checkpoint["batch_idx"]
            print(f"✔ Resumed from epoch {start_epoch}, batch {last_batch_idx}, step {global_step}")
        except RuntimeError as e:
            print(f"Warning: Checkpoint loading failed ({e}). Starting fresh.")
            # Optionally re-save tokenizer if the model was loaded successfully before
            # save_tokenizer(tokenizer, TOKENIZER_FILE)
    else:
        print("⏳ No checkpoint found — starting fresh.")
        # Save the newly built tokenizer
        save_tokenizer(tokenizer, TOKENIZER_FILE)


    # 9. Training Loop
    transformer.train()
    
    # Check if we are resuming from the end of an epoch (start_epoch == num_epochs)
    if start_epoch < num_epochs:
        print("\n-------------------------------\nSTARTING TRAINING\n-------------------------------")
        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            # Wrap train_loader with enumerate to get batch_idx
            progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, (src_data, tgt_data) in progress:
                
                # SKIP ALREADY COMPLETED BATCHES if resuming mid-epoch
                if epoch == start_epoch and batch_idx < last_batch_idx:
                    continue

                # Move data to the correct device
                src_data, tgt_data = src_data.to(device), tgt_data.to(device)

                optimizer.zero_grad()

                # Model forward: tgt[:, :-1] removes the last token (used for input)
                output = transformer(src_data, tgt_data[:, :-1]) 

                # Loss calculation: tgt[:, 1:] removes the first token (<bos>) (used for ground truth)
                # The output sequence (length L) predicts the next token at time t+1
                # The target sequence (length L+1) has tokens from t=1 to t=L+1
                loss = criterion(
                    # Flatten: (batch_size * (seq_len - 1), tgt_vocab_size)
                    output.contiguous().view(-1, tgt_vocab_size), 
                    # Flatten: (batch_size * (seq_len - 1),)
                    tgt_data[:, 1:].contiguous().view(-1) 
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                global_step += 1

                # Update progress bar
                progress.set_postfix({"loss": total_loss / (batch_idx - last_batch_idx + 1 if epoch == start_epoch else batch_idx + 1)})

                # SAVE CHECKPOINT EVERY N STEPS (e.g., every 200 steps)
                if global_step % 200 == 0:
                    torch.save({
                        "model_state": transformer.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                    }, latest_ckpt)
                    print(f"\n💾 Saved checkpoint at step {global_step}.")

            # Save checkpoint at end of epoch
            torch.save({
                "model_state": transformer.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "batch_idx": 0, # Reset batch index for the next epoch
                "global_step": global_step,
            }, latest_ckpt)

            print(f"\n🔥 Epoch {epoch+1} completed | Avg Loss: {total_loss / len(train_loader)}")
            last_batch_idx = 0  # reset after epoch

    else:
        print("Training completed in previous run. Skipping training loop.")
        
    # 10. Evaluation (on the held-out test set)
    test_ids = [t for t in test_tgt] # The target IDs are the actual sequence
    evaluate_model(transformer, test_src, test_tgt, tokenizer, device=device)

# --- Tokenizer Save/Load Functions (moved from notebook) ---

def save_tokenizer(tokenizer, path):
    """Saves the SinhalaTokenizer state to a JSON file."""
    data = {
        "w2i": tokenizer.w2i,
        "i2w": {str(k): v for k, v in tokenizer.i2w.items()}, # Save keys as strings for JSON
        "next_id": tokenizer.next_id
    }
    with open(path, "w", encoding="utf-8") as f:
        # indent=4 for human readability
        json.dump(data, f, ensure_ascii=False, indent=4) 
    print(f"Tokenizer saved to {path}")

def load_tokenizer(path):
    """Loads the SinhalaTokenizer state from a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer file not found at {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct the class
    tokenizer = SinhalaTokenizer()
    tokenizer.w2i = data["w2i"]
    # Convert keys back to integers (JSON loads dict keys as strings)
    tokenizer.i2w = {int(k): v for k, v in data["i2w"].items()}
    tokenizer.next_id = data["next_id"]
    return tokenizer

if __name__ == "__main__":
    # The original notebook installed sklearn. Let's ensure it's available.
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Installing required library: scikit-learn (for train_test_split)...")
        # You may need to run 'pip install scikit-learn' separately if permissions are an issue
        os.system("pip install scikit-learn")
        from sklearn.model_selection import train_test_split
        
    main()