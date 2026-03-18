import os
import shutil
import splitfolders

input_folder = 'animals10_data/raw-img'
output_folder = 'animals_splitted'

translate = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant",
    "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat",
    "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider"
}

splitfolders.ratio(input_folder, output=output_folder, seed=10, ratio=(0.8, 0.1, 0.1))
subsets = ['train', 'val', 'test']

for subset in subsets:
    subset_path = os.path.join(output_folder, subset)
    for old_name in os.listdir(subset_path):
        if old_name in translate:
            new_name = translate[old_name]
            old_dir = os.path.join(subset_path, old_name)
            new_dir = os.path.join(subset_path, new_name)

            if os.path.exists(new_dir):
                shutil.rmtree(new_dir)

            os.rename(old_dir, new_dir)

print(f"✅ Готово! Дані розбиті та перейменовані у папку: {output_folder}")
print("Структура:", os.listdir(os.path.join(output_folder, 'train')))