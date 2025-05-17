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
from rouge import Rouge

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants (adjust as needed)
MAX_LENGTH_ARTICLE = 512
MAX_LENGTH_SUMMARY = 128
WEIGHTS = {"bart": 0.40, "t5": 0.50, "pegasus": 0.10} # Example weights, can be tuned

# Load models and tokenizers globally to avoid reloading on each request
# This assumes the model directories exist. Error handling for missing directories should be added.
try:
    t5_tokenizer = T5Tokenizer.from_pretrained("Model/fine_tuned_t5_small")
    t5_model = T5ForConditionalGeneration.from_pretrained("Model/fine_tuned_t5_small").to(device)

    bart_tokenizer = BartTokenizer.from_pretrained("Model/fine_tuned_bart_cosine_3")
    bart_model = BartForConditionalGeneration.from_pretrained("Model/fine_tuned_bart_cosine_3").to(device)

    pegasus_tokenizer = PegasusTokenizer.from_pretrained("Model/fine_tuned_pegasus_custom")
    pegasus_model = PegasusForConditionalGeneration.from_pretrained("Model/fine_tuned_pegasus_custom").to(device)
except Exception as e:
    print(f"Error loading models: {e}")
    t5_tokenizer, t5_model = None, None
    bart_tokenizer, bart_model = None, None
    pegasus_tokenizer, pegasus_model = None, None


# Function to list model directories and add Ensemble option
def list_model_directories(model_base_path="Model"):
    model_paths = [os.path.join(model_base_path, d) for d in os.listdir(model_base_path) if os.path.isdir(os.path.join(model_base_path, d))]
    
    # Add Ensemble option if all required models are present
    required_for_ensemble = ["fine_tuned_t5_small", "fine_tuned_bart_cosine_3", "fine_tuned_pegasus_custom"]
    if all(os.path.isdir(os.path.join(model_base_path, d)) for d in required_for_ensemble):
        model_paths.append("Ensemble")

    return model_paths

# Function to generate summary from a single model
def generate_single_summary(model_name, article):
    model, tokenizer = None, None
    if model_name == "t5":
        model, tokenizer = t5_model, t5_tokenizer
    elif model_name == "bart":
        model, tokenizer = bart_model, bart_tokenizer
    elif model_name == "pegasus":
        model, tokenizer = pegasus_model, pegasus_tokenizer
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
