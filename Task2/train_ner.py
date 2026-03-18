import argparse
import json
import os
import torch
from ner import AnimalNER
from NER_data_generator import generate_and_save_dataset

def main():
    parser = argparse.ArgumentParser(description="Train Transformer-based NER model for Animals")

    parser.add_argument("--data_json", type=str, default="animal_ner_data_train.json", help="Path to synthetic dataset")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Base transformer model")
    parser.add_argument("--output_dir", type=str, default="animal_ner_model",
                        help="Directory to save the trained model")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for BERT")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("-" * 30)
    if device.type == 'cuda':
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("Training on CPU")
    print("-" * 30)

    if not os.path.exists(args.data_json):
        animals_list = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse',
                        'sheep', 'spider', 'squirrel']
        generate_and_save_dataset(animals_list, n_per_class=300, goal='train')

    with open(args.data_json, "r", encoding="utf-8") as f:
        training_data = json.load(f)

    ner_model = AnimalNER(model_name=args.model_name, device=device)

    ner_model.train(
        json_data=training_data,
        output_dir=args.output_dir,
        epochs=args.epochs
    )

    ner_model.save_model(args.output_dir)

if __name__ == "__main__":
    main()