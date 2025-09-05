import argparse
import torch

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT model for ticket routing.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pre-trained BERT model name.")
    parser.add_argument("--output_dir", type=str, default="../models/ticket_classifier_dual", help="Directory to save the trained model.")    
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    
    args = parser.parse_args()
    train(args)