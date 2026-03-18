import argparse
import json
import random

def generate_ner_dataset(animals, n_per_class=50, goal='train'):
    adjectives = ['big', 'small', 'beautiful', 'lovely', 'scary', 'wild', 'cute',
                  'fast', 'slow', 'fed',  'large', 'adult', 'healthy', 'young', 'mature', '']
    templates_train = [
        "There is a {adj} {animal} in the picture",
        "I can see a {adj} {animal} over there",
        "I can see a {adj} {animal} in the picture",
        "Is there a {adj} {animal} in the picture?",
        "Can I see a {adj} {animal} in the picture?",
        "Look at that {adj} {animal}",
        "I love this {adj} {animal}",
        "Is that a {adj} {animal}?",
        "Suddenly, a {adj} {animal} appeared"
    ]

    templates_test = [
        "There is a {adj} {animal} in the picture",
        "I can see a {adj} {animal} over there",
        "I can see a {adj} {animal} in the picture",
        "Is there a {adj} {animal} in the picture?",
        "Can I see a {adj} {animal} in the picture?",
        "Look at that {adj} {animal}",
        "I love this {adj} {animal}",
        "I believe a {adj} {animal} over there",
        "I believe I can see a {adj} {animal} over there",
        "It seems to me there is {adj} {animal} in the picture",
        "Somebody told me there is {adj} {animal} in the picture"
    ]

    dataset = []

    for animal in animals:
        for _ in range(n_per_class):
            if goal == 'train':
                template = random.choice(templates_train)
            else:
                goal = 'test'
                template = random.choice(templates_test)

            adj = random.choice(adjectives)

            sentence_text = template.format(adj=adj, animal=animal).replace("  ", " ").strip()
            tokens = sentence_text.split()

            tags = []
            for token in tokens:
                clean_token = token.lower().strip("!?. ,")
                if clean_token == animal.lower():
                    tags.append(1)
                else:
                    tags.append(0)

            dataset.append({
                "tokens": tokens,
                "ner_tags": tags
            })

    random.shuffle(dataset)
    return dataset

def generate_and_save_dataset(animals_list, n_per_class=300, goal='train'):
    ner_data = generate_ner_dataset(animals_list, n_per_class=n_per_class, goal=goal)

    with open(f"animal_ner_data_{goal}.json", "w", encoding="utf-8") as f:
        json.dump(ner_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train/test data for NER")

    parser.add_argument("--n_per_class", type=int, default=300, help="Number of sentences per class")
    parser.add_argument("--goal", type=str, default='train', help="Data for train or test")

    args = parser.parse_args()

    animals_list = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    generate_and_save_dataset(animals_list, n_per_class=args.n_per_class, goal=args.goal)



