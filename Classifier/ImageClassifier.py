import os
import cv2
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from shutil import copyfile


class CLASSIFIER:
    
    def __init__(self):
        pass
    
    def load_and_preprocess_image(self,img_path, target_size=(224, 224)):
        img = load_img(img_path, target_size=target_size)
        print(img)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def extract_features(self,img_path,model):
        img_data = self.load_and_preprocess_image(img_path)
        vgg16_feature = model.predict(img_data)
        return vgg16_feature.flatten()
    
    def get_similarity(self,feature1,feature2):
        feature1 = feature1.reshape(1,-1)
        feature2 = feature2.reshape(1,-1)
        similarity = cosine_similarity(feature1,feature2)
        return similarity
    
    def find_and_group_similar_images(self,
                                      sample_image_folder,
                                      dataset_folder,
                                      threshold=0.85,
                                      output_folder='output'):
        
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            
        model = VGG16(weights='imagenet', include_top=False)
        for sample_image_filename in os.listdir(sample_image_folder):
            if sample_image_filename.endswith(('.jpg','.png','.jpeg')):
                sample_image_path = os.path.join(
                    sample_image_folder,
                    sample_image_filename
                )
                
                sample_output_folder = os.path.join(output_folder,os.path.splitext(sample_image_filename)[0])
                
                if not os.path.exists(sample_output_folder):
                    os.mkdir(sample_output_folder)
                    
                sample_image_feature = self.extract_features(sample_image_path,model)
                
                for dataset_image_filename in os.listdir(dataset_folder):
                    if dataset_image_filename.endswith(('.jpg','.png','.jpeg')):
                        dataset_image_path = os.path.join(
                            dataset_folder,
                            dataset_image_filename
                        )
                        
                        current_feature = self.extract_features(dataset_image_path,model)
                        similarty = self.get_similarity(sample_image_feature,current_feature)
                        
                        if similarty > threshold:
                            output_path = os.path.join(sample_output_folder,dataset_image_filename)
                            copyfile(dataset_image_path,output_path)
                            print('Found similar image {} and {} with similarity {}'.format(
                                sample_image_filename,
                                dataset_image_filename,
                                similarty
                            ))