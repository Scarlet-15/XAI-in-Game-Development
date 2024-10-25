import cv2 as cv
from ultralytics import solutions
from scipy.spatial.distance import cosine
import numpy as np

def calculate_unambiguity(hm, similar_image_pairs):
    unambiguity_scores = []

    for img1, img2 in similar_image_pairs:
        heatmap1 = hm.generate_heatmap(img1)
        heatmap2 = hm.generate_heatmap(img2)
        #cv.imshow('hm1',heatmap1)
        #cv.imshow('hm2',heatmap2)
        #cv.waitKey(0)

        # Flatten heatmaps
        flat_map1 = heatmap1.flatten()
        flat_map2 = heatmap2.flatten()

        # Calculate cosine similarity
        similarity = 1 - cosine(flat_map1, flat_map2)
        unambiguity_scores.append(similarity)

    return np.mean(unambiguity_scores)


hm = solutions.Heatmap(model='best.pt',
                       colormap = cv.COLORMAP_VIRIDIS,
                       show = False)

img1 = cv.imread("test2.png")
img2 = cv.imread("test3.png")
img1_similar = cv.imread("test2similar.png")
img2_similar = cv.imread("test3similar.png")

img1 = cv.resize(img1, (640,640))
img2 = cv.resize(img2, (640,640))
img1_similar = cv.resize(img1_similar, (640,640))
img2_similar = cv.resize(img2_similar, (640,640))

similar_pairs = [(img1, img1_similar), (img2, img2_similar)]
unambiguity = calculate_unambiguity(hm,similar_pairs)
print(f"Unambiguity: {unambiguity:.3f}")