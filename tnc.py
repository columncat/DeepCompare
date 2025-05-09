import glob
import cv2
import csv
from typing import Dict, List, Tuple
from DeepComparer import DeepCompare
cvImage = cv2.typing.MatLike

def frame2time(frame_count:int, frame_rate:float):
    minutes:int = int((frame_count % (frame_rate * 3600)) // (frame_rate * 60))
    seconds:int = int((frame_count % (frame_rate * 60)) // frame_rate)
    return f'00:{minutes:02d}:{seconds:02d}'

def count_trim_number(video_path:str, reference_images_dict:Dict[str, List[cvImage]]) -> None:
    comparer:DeepCompare = DeepCompare(reference_images_dict)
    cap:cv2.VideoCapture = cv2.VideoCapture(video_path)

    frame_count:int = 0
    trim_count:int = 0
    last_detection:int = 0
    current_class:str = 'None'
    frame_rate:float = cap.get(cv2.CAP_PROP_FPS)
    
    video_title:str = video_path.split('\\')[-1].split('.')[0]
    
    file = open('count_trim.csv', 'a', newline='')
    writer = csv.writer(file)
    
    skip_frame:int = 9
    least_interval:float = 60.0 # average 70sec between cars, min 64sec
    probability_threshold:float = 0.90
    
    while True:
        start:int = cv2.getTickCount()
        
        ### where image_feeder will take place
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + skip_frame)  # Skip ahead 14 frames
        frame_count += skip_frame
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        ### where image_feeder will take place
        
        results:Dict[str, float] = comparer.compare_image(frame)
        
        detections:List[Tuple[str, float]] = [(class_name, probability) for class_name, probability in results.items() if probability > probability_threshold]
        class_name, probability = detections[0] if detections else (None, 0)
        if class_name:
            if (frame_count - last_detection > frame_rate * least_interval) or (frame_count < frame_rate * least_interval):
                last_detection:int = frame_count
                trim_count += 1
                current_class:str = class_name
                print(f'\rFound {class_name} at {frame2time(frame_count, frame_rate)}, {video_title}          ')
                writer.writerow([video_title, frame2time(frame_count, frame_rate), trim_count, class_name])
                
                # just skip here
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + frame_rate * least_interval)  # Skip ahead least_interval
                frame_count += frame_rate * least_interval
            else:
                last_detection:int = frame_count
        elif current_class != 'None':
            current_class:str = 'None'
        
        end:int = cv2.getTickCount()
        interval:float = (end - start) / cv2.getTickFrequency()
        
        print(f'\r{video_title} - {frame2time(frame_count, frame_rate)} | {interval * 1000:.2f}ms {1.0 / interval:.2f}fps', end='')
    cap.release()
    file.close()
    print('')

def main():
    reference_classes:List[str] = ['AR1', 'HZG', 'LJL', 'LFD']
    reference_images_dict:Dict[str, List[cvImage]] = {class_name: [] for class_name in reference_classes}
    
    for class_name in reference_classes:
        paths = glob.glob(f'../img/img_{class_name}_*.jpg')
        for path in paths:
            img:cvImage = cv2.imread(path)
            reference_images_dict[class_name].append(img)
    
    # linear processing
    video_files = glob.glob('../vid/*.mp4')
    for video_path in video_files:
        count_trim_number(video_path, reference_images_dict)

if __name__=='__main__':
    main()