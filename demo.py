import gradio as gr
import torch
import os
import numpy as np
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    PegasusTokenizer, PegasusForConditionalGeneration
)
from nltk.tokenize import sent_tokenize
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    PegasusTokenizer, PegasusForConditionalGeneration
)
from nltk.tokenize import sent_tokenize
from rouge import Rouge

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants (adjust as needed)
MAX_LENGTH_ARTICLE = 512
MAX_LENGTH_SUMMARY = 128
HIDDEN_DIM = 512 # From notebook
ENC_LAYERS = 3 # From notebook
DEC_LAYERS = 3 # From notebook
ENC_HEADS = 8 # From notebook
DEC_HEADS = 8 # From notebook
ENC_PF_DIM = 512 # From notebook
DEC_PF_DIM = 512 # From notebook
ENC_DROPOUT = 0.1 # From notebook
DEC_DROPOUT = 0.1 # From notebook
WEIGHTS = {"bart": 0.50, "t5": 0.45, "pegasus": 0.05} # Example weights, can be tuned

# Define Transformer model components (from Encoder_Decoder_Attention.ipynb)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=512):
        super().__init__()

        self.device = device

        self.embedding = nn.Embedding(vocab_size, hid_dim)
        self.pos_encoder = PositionalEncoding(hid_dim, dropout, max_length)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=n_heads,
            dim_feedforward=pf_dim,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask=None):
        src = src.transpose(0, 1)

        embedded = self.dropout(self.embedding(src) * self.scale)
        src = self.pos_encoder(embedded)

        src_key_padding_mask = src_mask if src_mask is not None else None

        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        return encoder_output.transpose(0, 1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=128):
        super().__init__()

        self.device = device

        self.embedding = nn.Embedding(vocab_size, hid_dim)
        self.pos_encoder = PositionalEncoding(hid_dim, dropout, max_length)

        decoder_layers = nn.TransformerDecoderLayer(
            d_model=hid_dim,
            nhead=n_heads,
            dim_feedforward=pf_dim,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, n_layers)

        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        trg = trg.transpose(0, 1)
        memory = memory.transpose(0, 1)

        embedded = self.dropout(self.embedding(trg) * self.scale)
        trg = self.pos_encoder(embedded)

        decoder_output = self.transformer_decoder(
            trg,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        output = self.fc_out(decoder_output)

        return output.transpose(0, 1)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device
        # Assuming tokenizer is available globally or passed
        self.sos_idx = transformer_tokenizer.bos_token_id 
        self.eos_idx = transformer_tokenizer.eos_token_id 


    def make_src_mask(self, src):
        src_mask = (src == self.pad_idx)
        return src_mask

    def make_trg_mask(self, trg):
        trg_key_padding_mask = (trg == self.pad_idx)
        trg_len = trg.shape[1]
        tgt_mask = torch.triu(torch.ones((trg_len, trg_len), device=self.device), diagonal=1).bool()
        return tgt_mask, trg_key_padding_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        tgt_mask, tgt_key_padding_mask = self.make_trg_mask(trg)

        encoder_output = self.encoder(src, src_mask)
        output = self.decoder(
            trg=trg,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_mask
        )

        return output

    def generate(self, input_ids, attention_mask=None, max_length=128, num_beams=4, length_penalty=2.0, early_stopping=True):
        batch_size = input_ids.shape[0]

        encoder_output = self.encoder(input_ids, (input_ids == self.pad_idx))

        decoder_input = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * self.sos_idx

        if num_beams > 1:
             # Simplified beam search for demo - might need full implementation from notebook
             # For now, fallback to greedy if num_beams > 1
             print("Beam search not fully implemented in demo. Falling back to greedy.")
             num_beams = 1


        if num_beams == 1:
            return self._generate_greedy(
                encoder_output=encoder_output,
                encoder_mask=(input_ids == self.pad_idx),
                start_token_id=self.sos_idx,
                end_token_id=self.eos_idx,
                max_length=max_length
            )
        else:
             # Placeholder for beam search if needed later
             return torch.empty(0) # Return empty tensor for now


    def _generate_greedy(self, encoder_output, encoder_mask, start_token_id, end_token_id, max_length):
        batch_size = encoder_output.shape[0]

        decoder_input = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * start_token_id

        completed_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_length - 1):
            tgt_mask, tgt_key_padding_mask = self.make_trg_mask(decoder_input)

            decoder_output = self.decoder(
                trg=decoder_input,
                memory=encoder_output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=encoder_mask
            )

            next_token_logits = decoder_output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            completed_sequences = completed_sequences | (next_token.squeeze(-1) == end_token_id)
            if completed_sequences.all():
                break

        return decoder_input


# Load models and tokenizers globally to avoid reloading on each request
# This assumes the model directories exist. Error handling for missing directories should be added.
try:
    t5_tokenizer = T5Tokenizer.from_pretrained("Model/fine_tuned_t5_small")
    t5_model = T5ForConditionalGeneration.from_pretrained("Model/fine_tuned_t5_small").to(device)

    bart_tokenizer = BartTokenizer.from_pretrained("Model/fine_tuned_bart_base")
    bart_model = BartForConditionalGeneration.from_pretrained("Model/fine_tuned_bart_base").to(device)

    pegasus_tokenizer = PegasusTokenizer.from_pretrained("Model/fine_tuned_pegasus_custom")
    pegasus_model = PegasusForConditionalGeneration.from_pretrained("Model/fine_tuned_pegasus_custom").to(device)

    # Load the transformer_scratch_best_model
    transformer_tokenizer = BartTokenizer.from_pretrained("Model/transformer_scratch_best_model")
    # Need vocab size from tokenizer to initialize the model
    transformer_vocab_size = len(transformer_tokenizer)
    transformer_pad_idx = transformer_tokenizer.pad_token_id

    transformer_encoder = Encoder(transformer_vocab_size, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, MAX_LENGTH_ARTICLE)
    transformer_decoder = Decoder(transformer_vocab_size, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device, MAX_LENGTH_SUMMARY)
    transformer_model = Seq2SeqTransformer(transformer_encoder, transformer_decoder, transformer_pad_idx, device)
    transformer_model.load_state_dict(torch.load("Model/transformer_scratch_best_model/best_transformer_model.pt", map_location=device))
    transformer_model.to(device)


except Exception as e:
    print(f"Error loading models: {e}")
    t5_tokenizer, t5_model = None, None
    bart_tokenizer, bart_model = None, None
    pegasus_tokenizer, pegasus_model = None, None
    transformer_tokenizer, transformer_model = None, None


# Function to list model directories and add Ensemble option
# Function to list model directories and add Ensemble option
def list_model_directories(model_base_path="Model"):
    # Get only directory names, not full paths
    model_names = [d for d in os.listdir(model_base_path) if os.path.isdir(os.path.join(model_base_path, d))]

    # Add Ensemble option if all required models are present
    required_for_ensemble = ["fine_tuned_t5_small", "fine_tuned_bart_base", "fine_tuned_pegasus_custom"]
    if all(os.path.isdir(os.path.join(model_base_path, d)) for d in required_for_ensemble):
        model_names.append("Ensemble")

    return model_names

# Function to generate summary from a single model
def generate_single_summary(model_name, article):
    model, tokenizer = None, None
    if model_name == "t5":
        model, tokenizer = t5_model, t5_tokenizer
    elif model_name == "bart":
        model, tokenizer = bart_model, bart_tokenizer
    elif model_name == "pegasus":
        model, tokenizer = pegasus_model, pegasus_tokenizer
    elif model_name == "transformer_scratch_best_model":
        model, tokenizer = transformer_model, transformer_tokenizer
    else:
        return "Invalid model name"

    if model is None or tokenizer is None:
        return f"Model {model_name} not loaded."

    model.eval()
    with torch.no_grad():
        if model_name == "t5":
            input_text = "summarize: " + article
            inputs = tokenizer(
                input_text,
                max_length=MAX_LENGTH_ARTICLE,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_LENGTH_SUMMARY,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        elif model_name == "transformer_scratch_best_model":
             inputs = tokenizer(
                article,
                max_length=MAX_LENGTH_ARTICLE,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
             output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_LENGTH_SUMMARY,
                num_beams=1, # Use greedy search as implemented in the notebook
                length_penalty=2.0,
                early_stopping=True
            )
        else:  # BART or Pegasus
            inputs = tokenizer(
                article,
                max_length=MAX_LENGTH_ARTICLE,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_LENGTH_SUMMARY,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                decoder_start_token_id=tokenizer.pad_token_id
            )
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

# Function to post-process summary
def post_process_summary(summary, tokenizer, max_length=MAX_LENGTH_SUMMARY):
    sentences = sent_tokenize(summary)
    unique_sentences = list(dict.fromkeys(sentences))  # Remove duplicates
    tokenized = tokenizer.encode(" ".join(unique_sentences), truncation=True, max_length=max_length)
    return tokenizer.decode(tokenized, skip_special_tokens=True)

# Ensemble function
def ensemble_summaries(article):
    if t5_model is None or bart_model is None or pegasus_model is None:
        return "Ensemble models not loaded."

    # Generate summaries from each model
    bart_summary = generate_single_summary("bart", article)
    t5_summary = generate_single_summary("t5", article)
    pegasus_summary = generate_single_summary("pegasus", article)

    # List of summaries
    summaries = {
        "bart": bart_summary,
        "t5": t5_summary,
        "pegasus": pegasus_summary
    }

    # Calculate ROUGE-L scores to select the best summary
    candidates = [summaries["bart"], summaries["t5"], summaries["pegasus"]]
    rouge = Rouge()
    scores = []

    for i, cand in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i]
        # Calculate average ROUGE-L score against other summaries
        if others:  # Ensure there are other summaries to compare against
            try:
                rouge_scores = rouge.get_scores([cand] * len(others), others, avg=True)
                rouge_l_score = rouge_scores["rouge-l"]["f"]
                weighted_score = rouge_l_score * WEIGHTS[list(summaries.keys())[i]]
                scores.append(weighted_score)
            except ValueError: # Handle cases where a summary might be empty or cause Rouge error
                 scores.append(0.0)
        else:
            scores.append(0.0)  # If no other summaries, assign a score of 0

    # Select the summary with the highest score
    best_idx = np.argmax(scores)
    best_summary = candidates[best_idx]

    # Post-process
    # Use BART tokenizer for post-processing as it was used in the notebook example
    final_summary = post_process_summary(best_summary, bart_tokenizer)
    return final_summary

# Function to chunk text and summarize
def chunk_and_summarize(text, model_path):
    words = text.split()
    if len(words) <= MAX_LENGTH_ARTICLE:
        if model_path == "Ensemble":
            return ensemble_summaries(text)
        else:
            model_type = None
            if "t5" in model_path.lower():
                model_type = "t5"
            elif "bart" in model_path.lower():
                model_type = "bart"
            elif "pegasus" in model_path.lower():
                model_type = "pegasus"
            elif "transformer_scratch_best_model" in model_path.lower():
                 model_type = "transformer_scratch_best_model"
            else:
                return "Could not determine model type from path."
            return generate_single_summary(model_type, text)
    else:
        # Simple chunking by splitting words
        chunks = []
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= MAX_LENGTH_ARTICLE:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        summaries = []
        for chunk in chunks:
            if model_path == "Ensemble":
                summary = ensemble_summaries(chunk)
            else:
                model_type = None
                if "t5" in model_path.lower():
                    model_type = "t5"
                elif "bart" in model_path.lower():
                    model_type = "bart"
                elif "pegasus" in model_path.lower():
                    model_type = "pegasus"
                elif "transformer_scratch_best_model" in model_path.lower():
                    model_type = "transformer_scratch_best_model"
                else:
                    return "Could not determine model type from path."
                summary = generate_single_summary(model_type, chunk)
            summaries.append(summary)

        # Concatenate summaries (simple concatenation)
        return " ".join(summaries)


# Summarization function
def summarize_text(text, model_path):
    if not text:
        return "", 0, 0
    if not model_path:
        return "Please select a model.", 0, 0

    input_length = len(text.split())
    
    summary = chunk_and_summarize(text, model_path)
    
    summary_length = len(summary.split())

    return summary, input_length, summary_length


# List available models
available_models = list_model_directories()
if not available_models:
    available_models = ["No models found in Model/ directory"]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Text Summarization")
    with gr.Row():
        text_input = gr.Textbox(label="Enter Text Here", lines=10)
        text_output = gr.Textbox(label="Summary", lines=10)
    with gr.Row():
        input_length_output = gr.Textbox(label="Input Length (words)", interactive=False)
        summary_length_output = gr.Textbox(label="Summary Length (words)", interactive=False)
    model_dropdown = gr.Dropdown(
        label="Select Model",
        choices=available_models,
        value=available_models[0] if available_models and available_models[0] != "No models found in Model/ directory" else None
    )
    summarize_button = gr.Button("Summarize")

    summarize_button.click(
        summarize_text,
        inputs=[text_input, model_dropdown],
        outputs=[text_output, input_length_output, summary_length_output]
    )

if __name__ == "__main__":
    demo.launch()
