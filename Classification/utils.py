import os
import shutil

<<<<<<< HEAD
path = "/home/ananya/Documents/VR/assignments/group2_project_data/images/yolo/"
folders = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) ]
train_path = "/home/ananya/Documents/VR/assignments/group2_project_data/images/yolo/train"
test_path = "/home/ananya/Documents/VR/assignments/group2_project_data/images/yolo/val"
=======

path = "yolo_images/"
folders = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) ]
train_path = "yolo_images/train"
test_path = "yolo_images/val"
>>>>>>> 1b7cd3ec376c9eaf3d9a5f61be516a8252b9aa12

for folder in folders:
    images = []
    for filename in os.listdir(path+folder):
        source = os.path.join(path+folder, filename)
        images.append(source)

    num_images = len(images)
    train_images = images[ :(num_images - int(num_images/10))]
    test_images = images[(num_images - int(num_images/10)):]

    train_des = os.path.join(train_path, folder)
    os.makedirs(train_des)
    for filename in train_images:
        shutil.move(filename, train_des)

    test_des = os.path.join(test_path, folder)
    os.makedirs(test_des)
    for filename in test_images:
        shutil.move(filename, test_des)
