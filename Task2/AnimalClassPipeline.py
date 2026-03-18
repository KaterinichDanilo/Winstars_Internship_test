from PIL import Image
from torchvision import transforms

class AnimalMultiModalPipeline:
    def __init__(self, cv_model, ner_model, class_names, device):
        """
        :param cv_model: An instance of class AnimalClassifier (ResNet50)
        :param ner_model: An instance of class AnimalNER (DistilBERT)
        :param class_names: list of classes names for cv model
        """
        self.cv_model = cv_model
        self.ner_model = ner_model
        self.class_names = class_names
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = device

    def predict(self, image_path, text_query):
        """
        :param image_path: path to image
        :param text_query: query for AnimalNER model
        :return: True if the animal in the text matches the animal in the image, False otherwise
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image).to(self.device)
        cv_indices = self.cv_model.predict(image)
        cv_label = self.class_names[cv_indices[0]]

        detected_entities = self.ner_model.predict(text_query)

        is_match = cv_label in detected_entities

        print(f"DEBUG: CV detected -> {cv_label}")
        print(f"DEBUG: NER detected -> {detected_entities}")
        print(f"RESULT: {'Match!' if is_match else 'No match.'}")

        return is_match