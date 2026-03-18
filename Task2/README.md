# Task 2. Named entity recognition + image classification

This repository contains a dual-model system designed to extract animal names from text and verify their presence in images. The project combines Named entity recognition (NER) and Computer Vision (CV) into a single functional pipeline.

### Models

Animal NER: A fine-tuned DistilBERT model that identifies 10 specific animal classes within any English sentence.

Image Classifier model: A fine-tuned **resnet50** model that classifies 10 animals.

### Implementation details

1. Named Entity Recognition (NER)
    Model: Based on distilbert-base-uncased for its optimal balance between speed and accuracy.
    Synthetic Data: Since no specific "Animals-10 NER" dataset existed, I developed a Template-based Generator to create 3,000+ labeled sentences with BIO-tagging (B-ANIMAL, I-ANIMAL, O).
    Tokenization: Handled sub-word tokenization challenges (e.g., splitting "butterfly" into sub-tokens) by aligning labels during the fine-tuning process.
2. MultiModal Pipeline Logic
    Input: Takes an image path and a text string.
    Vision Step: The AnimalClassifier (ResNet-50) predicts the animal in the image.
    Text Step: The AnimalNER extracts all animal entities from the text.
    Verification: The pipeline returns True only if the animal detected in the image is present in the list of animals extracted from the text.

## Usage
First download dataset
```python
from AnimalClassifier import AnimalClassifier
from ner import AnimalNER
from AnimalClassPipeline import AnimalMultiModalPipeline

# Create and train you model
model = AnimalClassifier(device, lr=lr)
model.train(train_loader, val_loader, epochs=epochs)
# Save trained model
model.save_model(model_path)

# Create and train ner model
ner_model = AnimalNER(model_name=model_name, device=device)

ner_model.train(
    json_data=training_data,
    output_dir=output_dir,
    epochs=epochs
)

ner_model.save_model(output_dir)

# Create pipeline and ake prediction
classes_name = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep',
                'spider', 'squirrel']

pipeline = AnimalMultiModalPipeline(model, ner_model, classes_name, device)

img_pass = 'test_images/cow.png'
text = 'Is there a cow or dog in this picture?'

result = pipeline.predict(img_pass, text)
print(result)
```