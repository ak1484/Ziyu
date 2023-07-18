import super_gradients
import os
import cv2
import os
yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco").cuda()
def save_image(path_image,image):
    test_path = '/home/ankit/yolonas/Ziyu/test1'
    os.makedirs(test_path, exist_ok=True)
    image_name = path_image.rsplit('/home/ankit/yolonas/Ziyu/Episode 001/', 1)[-1]
    test_path = os.path.join(test_path,image_name)
    image.save(test_path)
    
    folder_path = "/home/ankit/yolonas/Ziyu/test"
    os.makedirs(folder_path, exist_ok=True)
    old_path = os.path.join(test_path,"pred_0.jpg")
    new_path = os.path.join(folder_path, image_name)
    os.rename(old_path, new_path)

def predict_objects(image_path):
    image = cv2.imread(image_path)
    confidence_threshold = 0.75
    predictions = yolo_nas.predict(image,conf=confidence_threshold)
    save_image(image_path,predictions)
    
image_folder = "/home/ankit/yolonas/Ziyu/Episode 001"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    predict_objects(image_path)
