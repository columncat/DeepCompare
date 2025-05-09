# Copyright (c) 2024-2025 Pusan National University and Renault Korea
# Contributors: team ICU (Geonju Kim, Dongin Park, Taeju Park, Jinyeong Cho, Hyunsik Jeong)
# Organization: Pusan National University
# Cooperation: Renault Korea
# Project: Detecting Missing or Inappropriately assembled parts in Car trim process Via CCTV
# Date: 2024.09.01. ~ 2025.06.30.
# Description: This module processes video files to detect vehicles using DeepCompare.
# License: MIT License

import cv2
import csv
import os
import logging
from typing import Dict, List, Tuple, Optional
from DeepCompare import DeepCompare
from tqdm import tqdm
from data import load_reference_images, get_video_files

cvImage = cv2.typing.MatLike

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tnc_processing.log'),
        logging.StreamHandler()
    ]
)

def frame2time(frame_count: int, frame_rate: float) -> str:
    """Convert frame number to time format (MM:SS)."""
    hours: int = int(frame_count // (frame_rate * 3600))
    minutes: int = int((frame_count % (frame_rate * 3600)) // (frame_rate * 60))
    seconds: int = int((frame_count % (frame_rate * 60)) // frame_rate)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def count_trim_number(
    video_path: str, 
    reference_images_dict: Dict[str, List[cvImage]], 
    skip_frame: int = 60,
    least_interval: float = 45.0,
    probability_threshold: float = 0.90
) -> None:
    """
    Detect vehicles in a video and record results to a CSV file.
    
    Args:
        video_path: Path to the video file to process
        reference_images_dict: Dictionary of reference images
        skip_frame: Number of frames to skip between processing
        least_interval: Minimum interval between consecutive detections (seconds)
        probability_threshold: Minimum probability to consider as a detection
    """
    try:
        comparer: DeepCompare = DeepCompare(reference_images_dict)
        cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return
            
        frame_count: int = 0
        trim_count: int = 0
        last_detection: int = 0
        current_class: str = 'None'
        
        # Get frame rate
        frame_rate: float = cap.get(cv2.CAP_PROP_FPS)
        
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_title: str = os.path.basename(video_path).split('.')[0]
        
        logging.info(f"Started processing video: {video_title}, frame rate: {frame_rate:.2f}, total frames: {total_frames}")
        
        os.makedirs(os.path.dirname(os.path.abspath('count_trim.csv')), exist_ok=True)
        with open('count_trim.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Add header if CSV file is newly created
            if os.path.getsize('count_trim.csv') == 0:
                writer.writerow(['Video', 'Timestamp', 'Count', 'Class'])
            
            pbar = tqdm(total=total_frames, desc=f"Processing: {video_title}")
            
            while frame_count < total_frames:
                start: int = cv2.getTickCount()
                
                # Move and read frame
                if skip_frame == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    frame_diff = 1
                else:
                    next_frame = min(frame_count + skip_frame, total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                    frame_diff = next_frame - frame_count
                    frame_count = next_frame
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                
                # Image comparison and detection
                results: Dict[str, float] = comparer.compare_image(frame)
                
                detections: List[Tuple[str, float]] = [
                    (class_name, probability) 
                    for class_name, probability in results.items() 
                    if probability > probability_threshold
                ]
                
                class_name: Optional[str] = None
                probability: float = 0
                
                if detections:
                    class_name, probability = detections[0]
                
                if class_name:
                    if (frame_count - last_detection > frame_rate * least_interval) or (frame_count < frame_rate * least_interval):
                        last_detection = frame_count
                        trim_count += 1
                        current_class = class_name
                        timestamp = frame2time(frame_count, frame_rate)
                        
                        # Print log message with clear line breaks
                        log_message = f"{video_title}: {class_name} detected ({probability:.2f}) - time: {timestamp}, count: {trim_count}"
                        logging.info(log_message)
                        
                        writer.writerow([video_title, timestamp, trim_count, class_name])
                        
                        # Skip forward to maintain minimum interval
                        skip_to_frame = min(frame_count + int(frame_rate * least_interval), total_frames - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_to_frame)
                        pbar.update(skip_to_frame - frame_count)
                        frame_count = skip_to_frame
                    else:
                        last_detection = frame_count
                elif current_class != 'None':
                    current_class = 'None'
                
                end: int = cv2.getTickCount()
                interval: float = (end - start) / cv2.getTickFrequency()
                fps: float = 1.0 / interval if interval > 0 else 0
                
                # Update progress bar format - ensure time info isn't truncated
                pbar.set_postfix_str(f"Time: {frame2time(frame_count, frame_rate)}, FPS: {fps:.2f}")
                pbar.update(frame_diff + 1)
                
            pbar.close()
            
    except Exception as e:
        logging.exception(f"Error processing video: {video_path}, {str(e)}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        logging.info(f"Completed processing video: {video_title}, vehicles detected: {trim_count}")

def main():
    try:
        # Load reference images from data module
        reference_images_dict = load_reference_images()
        
        # Get video files from data module
        video_files = get_video_files()
        
        if not video_files:
            logging.warning("No video files to process")
            return
            
        logging.info(f"Starting to process {len(video_files)} video files")
        
        # Process videos sequentially
        for video_path in video_files:
            count_trim_number(video_path, reference_images_dict)
            
        logging.info("All videos processed")
        
    except Exception as e:
        logging.exception(f"Error running program: {str(e)}")

if __name__=='__main__':
    main()
