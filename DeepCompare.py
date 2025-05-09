# Author: team ICU (Geonju Kim, Dongin Park, Taeju Park, Jinyeong Cho, Hyunsik Jeong)
# Organization: Pusan National University
# Cooperation: Renault Korea
# Project: Detecting Missing or Inappropriately assembled parts in Car trim process Via CCTV
# Date: 2024.09.01. ~ 2025.06.30.
# Description: This module compares images using pretrained-DL model's embeddings to find and classify scenes.
# License: MIT License

import numpy as np
import logging
import cv2
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from collections import defaultdict
from scipy.special import softmax
from typing import Dict, List
cvImage = cv2.typing.MatLike

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepCompare:
    def __init__(self, reference_images_dict: Dict[str, List[cvImage]]) -> None:
        # Load device
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {self.device}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
        
        # Load MobileNetV2
        logging.info("Using backbone: MobileNetV2 1K_V1")
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model: nn.Sequential = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d((1, 1))
        ).to(self.device).eval()

        self.embeddings: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for class_name, images in reference_images_dict.items():
            for image in images:
                embedding: torch.Tensor = self.get_embedding(image)
                self.embeddings[class_name].append(embedding)

    def preprocess_image(self, image: cvImage) -> cvImage:
        """Apply ROI and CLAHE preprocessing."""
        height, width = image.shape[0], image.shape[1]
        img1: cvImage = image[0:int(2/3 * height), int(0.3 * width):int(0.5 * width)]
        img2: cvImage = image[0:int(2/3 * height), int(0.7 * width):int(0.9 * width)]
        img_concat: cvImage = np.concatenate((img1, img2), axis=1)

        # Convert to CIELAB color space
        lab = cv2.cvtColor(img_concat, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to Lightness channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
        cl = clahe.apply(l)
        
        # Merge and return to BGR
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def load_image(self, image: cvImage) -> torch.Tensor:
        img_clahe = self.preprocess_image(image)

        # BGR -> RGB 및 normalization
        resized_img: cvImage = cv2.resize(img_clahe, (224, 224))
        rgb_img: cvImage = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_tensor: torch.Tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float().div(255.0)
        mean_values = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std_values = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean_values) / std_values

        return img_tensor.unsqueeze(0).to(self.device)

    def get_embedding(self, image: cvImage) -> torch.Tensor:
        image_tensor: torch.Tensor = self.load_image(image)
        with torch.no_grad():
            features: torch.Tensor = self.model(image_tensor)
            embedding: torch.Tensor = features.view(features.size(0), -1)   # Flattening
        return embedding

    @staticmethod
    def cosine_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        emb1_normalized = torch.nn.functional.normalize(emb1.view(1, -1), p=2, dim=1)
        emb2_normalized = torch.nn.functional.normalize(emb2.view(1, -1), p=2, dim=1)
        cos_sim = torch.sum(emb1_normalized * emb2_normalized)
        return 1.0 - cos_sim.item()

    def compare_image(self, image: cvImage) -> Dict[str, float]:
        embedding: torch.Tensor = self.get_embedding(image)
        class_names: List[str] = list(self.embeddings.keys())
        min_distances: List[float] = []

        for class_name in class_names:
            class_distances: List[float] = [self.cosine_distance(embedding, emb) for emb in self.embeddings[class_name]]
            min_distances.append(min(class_distances))

        inv_distances: List[float] = [1.0 / dist if dist > 0 else float('inf') for dist in min_distances]
        inv_distances_array = np.array(inv_distances, dtype=np.float64)
        probabilities: np.ndarray = softmax(inv_distances_array)
        return {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
