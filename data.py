import os
import glob
import cv2
import logging
from typing import Dict, List, Optional

cvImage = cv2.typing.MatLike

REFERENCE_CLASSES = ['Input', 'your', 'classes', 'here']
IMAGE_DIR = 'path/to/your/reference/images/img_CLS_*.jpg'   # CLS will be replaced with class names
VIDEO_DIR = 'path/to/your/video/files'
VIDEO_EXTENSIONS = ['your', 'video', 'extensions', 'here']  # e.g., ['mp4', 'avi']

def load_reference_images() -> Dict[str, List[cvImage]]:
    """
    Load reference images for all classes.
    
    Returns:
        Dict mapping class names to lists of reference images
    """
    reference_images_dict: Dict[str, List[cvImage]] = {class_name: [] for class_name in REFERENCE_CLASSES}
    
    # Load reference images
    for class_name in REFERENCE_CLASSES:
        paths = glob.glob(IMAGE_DIR.replace('CLS', class_name))
        if not paths:
            logging.warning(f"No reference images found for class {class_name}")
        
        for path in paths:
            if not os.path.exists(path):
                logging.warning(f"Image file not found: {path}")
                continue
                
            img: Optional[cvImage] = cv2.imread(path)
            if img is None:
                logging.warning(f"Could not read image file: {path}")
                continue
                
            reference_images_dict[class_name].append(img)
            
        logging.info(f"Loaded {len(reference_images_dict[class_name])} reference images for class {class_name}")
    
    return reference_images_dict

def get_video_files() -> List[str]:
    """
    Get a list of video files to process.
    
    Returns:
        List of video file paths
    """
    video_files = []
    
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(f'{VIDEO_DIR}/*.{ext}'))
    
    if not video_files:
        logging.warning("No video files to process")
    
    return video_files