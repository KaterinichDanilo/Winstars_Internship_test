import argparse
import json
import os
import torch
from NER_data_generator import generate_and_save_dataset
from ner import AnimalNER

def main():
    parser = argparse.ArgumentParser(description="Inference for trained Animal NER model")
    parser.add_argument("--test_data", type=str, default="animal_ner_data_test.json", help="Path to test dataset")
    parser.add_argument("--model_path", type=str, default="animal_ner_model", help="Path to saved NER model")
    args = parser.parse_args()

    if not os.path.exists(args.test_data):
        animals_list = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse',
                        'sheep', 'spider', 'squirrel']
        generate_and_save_dataset(animals_list, n_per_class=100, goal='test')

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model = AnimalNER(model_name=args.model_path, device=device)

    correct_count = 0
    for sample in test_samples:
        sentence = " ".join(sample["tokens"])
        expected = [sample["tokens"][i] for i, tag in enumerate(sample["ner_tags"]) if tag == 1]

        detected = ner_model.predict(sentence)

        is_correct = set(expected) == set(detected)
        if is_correct:
            correct_count += 1

    accuracy = correct_count / len(test_samples)
    print(f"Accuracy: {accuracy:.4f}")

    print(ner_model.predict('Is there big elephant and black horse in the picture?'))


if __name__ == "__main__":
    main()