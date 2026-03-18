import kagglehub
import shutil
import os

# Download latest version
downloaded_path = kagglehub.dataset_download("alessiocorrado99/animals10")

print("Path to dataset files:", downloaded_path)
target_path = os.path.join(os.getcwd(), 'animals10_data')

if not os.path.exists(target_path):
    print(f"Moving dataset to: {target_path}")
    shutil.move(downloaded_path, target_path)
    print("Done!")
else:
    print(f"Dataset already exists at: {target_path}")

print("Current dataset path:", target_path)

translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant",
             "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat",
             "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel",
             "dog": "cane", "cavallo": "horse", "elephant" : "elefante",
             "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto",
             "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}
