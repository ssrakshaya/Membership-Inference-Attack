#Importing pytorch - which is used to run the model and maange tensors (on the CPU and GPU)
#transformers uses the hugging face library, auto tokenizer prepares text for models
#Auto model loads a language model 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


"""
This function takes a string of text 
and computes the log probabilities of each token the model predicts. 
Why? Because low log-probs indicate model "surprise," which is key to min‑k%++.
"""
def calculate_log_probs(text, model, tokenizer, device='cpu'):
    """
    Given an input text, this function:
    - Tokenizes the text.
    - Runs it through the language model to get token logits.
    - Computes the log probabilities for each token.
    Returns a list of log probabilities for each token (excluding the first).
    """
    model.eval() #sets model to evaluation mode
    # Tokenizes input text into IDs and move to the correct device (CPU or GPU)
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    
    #runs the model without tracking gradients
    with torch.no_grad():
        # The model predicts the next token probabilities and computes loss
        outputs = model(inputs, labels=inputs)
        #Also calculates the loss and raw model outputs (logits), 
        # #which are scores for each token prediction.
        loss, logits = outputs[:2]
    
    # Convert logits to log-probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Exclude the first token (no prediction for it) and collect log-probs for the rest
    token_ids = inputs[0][1:]
    token_log_probs = [log_probs[0, i, token_id].item() for i, token_id in enumerate(token_ids)]
    return token_log_probs

def min_k_percent_plus(token_log_probs, k_percent=0.1):
    """
    Implements the min-k%++ metric.
    - Sorts the token log probabilities in ascending order.
    - Selects the bottom k% (the tokens the model is least confident about).
    - Returns the negative mean of these selected log probabilities.
      (A higher score indicates more 'surprise' on the least confident tokens.)
    """
    k = int(len(token_log_probs) * k_percent)
    if k == 0:
        return None
    sorted_log_probs = np.sort(token_log_probs)
    bottom_k = sorted_log_probs[:k]
    return -np.mean(bottom_k)

def main():
    # Choose a free model from Hugging Face (this one doesn't use any paid API)
    model_name = "distilgpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Example texts for your experiments
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Membership inference attacks reveal private training data.",
        "This is a sentence that is likely from a training dataset.",
        "Another random sentence to test the model."
    ]
    
    # Evaluate each text using min-k%++ for various percentages
    for text in test_texts:
        token_log_probs = calculate_log_probs(text, model, tokenizer, device=device)
        scores = {}
        for k_percent in [0.05, 0.1, 0.2]:
            score = min_k_percent_plus(token_log_probs, k_percent=k_percent)
            scores[f"min_{int(k_percent*100)}%"] = score
        print("Text:", text)
        print("Min-k++ scores:", scores)
        print("-" * 50)

if __name__ == "__main__":
    main()
