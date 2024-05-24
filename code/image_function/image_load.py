import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def resize_image(img_path, img_size=(150, 150)):
    # 'resize image to img_size'
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, img_size)
    return img

def resize_and_convert_to_grayscale(img_path, img_size=(150, 150)):
    # 'resize image to img_size and convert to grayscale'
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def load_image(food_path, food, image_list, target_list, except_list, img_size=(150, 150), pic_cnt=100, grayscale=False):
    # "Image load "
    for idx, img in enumerate(os.listdir(food_path)):
        if idx == pic_cnt:
            break
        img_path = os.path.join(food_path, img)
        img_type = img_path.split('.')[-1].lower()
        if img_type not in ['properties', 'csv']:
            if grayscale:
                img = resize_and_convert_to_grayscale(img_path, img_size)
            else:
                img = resize_image(img_path, img_size)
            if img is not None:
                image_list.append(img)
                target_list.append(food)
            else:
                except_list.append(img_path)

def load_and_resize_images(food_path_list, food_list, img_size=(150, 150), pic_cnt=100, grayscale=False):
    # "병렬계산을 통해서 리스트로"
    image_list = []
    target_list = []
    except_list = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_image, food_path, food, image_list, target_list, except_list, img_size, pic_cnt, grayscale)
                   for food_path, food in zip(food_path_list, food_list)]
        for future in tqdm(futures, desc="Processing images"):
            future.result()

    return np.array(image_list, dtype=np.float32), np.array(target_list), except_list