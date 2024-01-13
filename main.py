from Classifier.ImageClassifier import CLASSIFIER

# Path: main.py
classifier = CLASSIFIER()

SAMPLE_IMAGE_FOLDER = 'Data/Sample'
DATASET_FOLDER = 'Data/Dataset'
OUTPUT_FOLDER = 'Data/ClassifiedImages'
THRESHOLD = 0.5

classifier.find_and_group_similar_images(
    SAMPLE_IMAGE_FOLDER,
    DATASET_FOLDER,
    THRESHOLD,
    OUTPUT_FOLDER
)