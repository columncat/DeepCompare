import glob
import cv2
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import numpy as np
import os
from collections import defaultdict
from scipy.special import softmax
from typing import Dict, List, Optional
cvImage = cv2.typing.MatLike


class DeepCompare:
    def __init__(self, reference_images_dict:Dict[str, List[cvImage]]) -> None:
        self.device:torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device:{self.device}')
        
        self.model:nn.Sequential = nn.Sequential(
            *list(models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1]
        ).to(self.device).eval()
        
        self.embeddings:Dict[str, List[torch.Tensor]] = defaultdict(list)
        for class_name, images in reference_images_dict.items():
            for image in images:
                embedding:torch.Tensor = self.get_embedding(image)
                self.embeddings[class_name].append(embedding)
    
    def load_image(self, image:cvImage) -> torch.Tensor:
        # Applying Region Of Interest
        height, width = image.shape[0], image.shape[1]
        img1:cvImage = image[0:int(2/3 * height), int(0.3 * width):int(0.5 * width)]
        img2:cvImage = image[0:int(2/3 * height), int(0.7 * width):int(0.9 * width)]
        img_concat:cvImage = np.concatenate((img1, img2), axis=1)
        
        # Preprocessing for BGR image
        resized_img:cvImage = cv2.resize(img_concat, (224, 224))
        rgb_img:cvImage = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_tensor:torch.Tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float().div(255.0)
        mean_values = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std_values = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean_values) / std_values
        
        # Add batch dimension and move to device
        batched_img:torch.Tensor = img_tensor.unsqueeze(0).to(self.device)
        return batched_img
    
    def get_embedding(self, image:cvImage) -> torch.Tensor:
        image_tensor:torch.Tensor = self.load_image(image)
        with torch.no_grad():
            embedding:torch.Tensor = self.model(image_tensor)
        return embedding
    
    @staticmethod
    def cosine_distance(emb1:torch.Tensor, emb2:torch.Tensor) -> float:
        emb1_normalized = torch.nn.functional.normalize(emb1.view(1, -1), p=2, dim=1)
        emb2_normalized = torch.nn.functional.normalize(emb2.view(1, -1), p=2, dim=1)
        cos_sim = torch.sum(emb1_normalized * emb2_normalized)
        return 1.0 - cos_sim.item()
    
    def compare_image(self, image:cvImage) -> Dict[str, float]:
        embedding:torch.Tensor = self.get_embedding(image)
        class_names:List[str] = list(self.embeddings.keys())
        min_distances:List[float] = []
        
        for class_name in class_names:
            class_distances:List[float] = [self.cosine_distance(embedding, emb) for emb in self.embeddings[class_name]]
            min_distances.append(min(class_distances))
        inv_distances:List[float] = [1.0 / dist if dist > 0 else float('inf') for dist in min_distances]
        inv_distances_array = np.array(inv_distances, dtype=np.float64)
        probabilities:np.ndarray = softmax(inv_distances_array)
        return {class_name:float(prob) for class_name, prob in zip(class_names, probabilities)}
