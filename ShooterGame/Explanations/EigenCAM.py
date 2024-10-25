from ultralytics import YOLO
import cv2 as cv
from yolo_cam.eigen_cam import EigenCAM
import numpy as np
from scipy.spatial.distance import cosine
import torch

class EigenCAMMetricsCalculator:
    def __init__(self, model, target_layers):
        self.model = model
        self.cam = EigenCAM(model, target_layers, task='od')

    def preprocess_image(self, cv2_image):
        # Resize image
        resized_image = cv.resize(cv2_image, (640, 640))
        image_rgb = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        return image_tensor, resized_image

    def get_eigencam_map(self, cv2_image):
        # Generate EigenCAM
        grayscale_cam = self.cam(cv2_image)[0, :, :]
        return grayscale_cam

    def calculate_unambiguity(self, similar_image_pairs):
        unambiguity_scores = []

        for img1, img2 in similar_image_pairs:
            # Generate EigenCAM explanations
            eigencam_map1 = self.get_eigencam_map(img1)
            eigencam_map2 = self.get_eigencam_map(img2)

            # Flatten heatmaps
            flat_map1 = eigencam_map1.flatten()
            flat_map2 = eigencam_map2.flatten()

            # Calculate cosine similarity
            similarity = 1 - cosine(flat_map1, flat_map2)
            unambiguity_scores.append(similarity)

        return np.mean(unambiguity_scores)

test_image = cv.imread('test.png')
test_image = cv.resize(test_image, (640, 640))
img = test_image.copy()
img = np.float32(img) / 255

model = YOLO('best.pt')
target_layers = [model.model.model[-3]]
calculator = EigenCAMMetricsCalculator(model, target_layers)

img1 = cv.imread("test1.png")
img2 = cv.imread("test2.png")
img3 = cv.imread("test3.png")
img1_similar = cv.imread("test2similar.png")
img2_similar = cv.imread("test3similar.png")

img1 = cv.resize(img1, (640,640))
img2 = cv.resize(img2, (640,640))
img3 = cv.resize(img3, (640,640))
img1_similar = cv.resize(img1_similar, (640,640))
img2_similar = cv.resize(img2_similar, (640,640))


images = [img1, img2, img3]
similar_pairs = [(img1, img1_similar), (img2, img2_similar)]
unambiguity = calculator.calculate_unambiguity(similar_pairs)

print(f"Unambiguity: {unambiguity:.3f}")