from transformers import AutoTokenizer, AutoModelForCausalLM
#from transformers.utils import init_empty_weights
import torch
import numpy as np

# This function computes the log probabilities of each token in the input text.
def calculate_log_probs(text, model, tokenizer, device='cpu'):
    model.eval()  # Set model to evaluation mode (no training)
    
    # Tokenize the input text and move to device (CPU or GPU)
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)

    # Forward pass: model predicts its own input and returns loss + logits
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss, logits = outputs[:2]

    # Convert raw logits to log-probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Extract token IDs, excluding the first one (since it has no prediction)
    token_ids = inputs[0][1:]
    
    # Collect log probabilities for each actual token in the input
    token_log_probs = [
        log_probs[0, i, token_id].item() for i, token_id in enumerate(token_ids)
    ]
    return token_log_probs

# This function implements the core idea of min-k%++ attack.
def min_k_percent_plus(token_log_probs, k_percent=0.1):
    # Determine how many tokens make up the bottom-k%
    k = int(len(token_log_probs) * k_percent)
    if k == 0:
        return None  # Avoid dividing by zero

    # Sort token log-probs from lowest to highest (most uncertain to least)
    sorted_log_probs = np.sort(token_log_probs)

    # Select the bottom-k% log-probabilities
    bottom_k = sorted_log_probs[:k]

    # Return the negative mean of those bottom-k% log-probs (higher = more memorization)
    return -np.mean(bottom_k)

# Legacy: Perplexity (for optional use later)
def calculatePerplexity(text, model, tokenizer, device):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    input_ids_processed = input_ids[0][1:]
    all_prob = [probabilities[0, i, token_id].item() for i, token_id in enumerate(input_ids_processed)]
    return torch.exp(loss).item(), all_prob, loss.item()

# Evaluate a list of text examples and return their scores
def evaluate_data(text_list, model, tokenizer, device='cpu', k_percents=[0.1]):
    results = []
    for text in text_list:
        token_log_probs = calculate_log_probs(text, model, tokenizer, device)
        score_dict = {'text': text}
        for k in k_percents:
            score = min_k_percent_plus(token_log_probs, k_percent=k)
            score_dict[f'min_k_{int(k*100)}%'] = score
        results.append(score_dict)
    return results

# Main execution
def main():
    model_name = "distilgpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Example test set (can replace with your real dataset later)
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Membership inference attacks reveal private training data.",
        "This sentence was likely seen during training.",
        "Deep learning models can memorize text."
    ]

    # Evaluate using min-k%++
    results = evaluate_data(test_sentences, model, tokenizer, device, k_percents=[0.05, 0.1, 0.2])
    
    for result in results:
        print(f"\nText: {result['text']}")
        for key, val in result.items():
            if key != "text":
                print(f"  {key}: {val:.4f}")

if __name__ == "__main__":
    main()