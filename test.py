import os
import csv
import cv2

path = '/opt/carnd_p3/data/'

imgs = []
with open(path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skip first row
    next(reader)
    for img in reader:
        imgs.append(img)
    
for img in imgs:
    # center image
    name = '/opt/carnd_p3/data/IMG/'+img[0].split('/')[-1]
    center_image = cv2.imread(name)
    # convert to RGB
    center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('examples/center_image.jpg', center_image_rgb)
    cv2.imwrite('examples/center_image_flip.jpg', cv2.flip(center_image_rgb, 1))
    
    l_name = '/opt/carnd_p3/data/IMG/'+img[1].split('/')[-1]
    left_image = cv2.imread(l_name)
    # convert to RGB
    left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('examples/left_image.jpg', left_image_rgb)
    cv2.imwrite('examples/left_image_flip.jpg', cv2.flip(left_image_rgb, 1))
    
    r_name = '/opt/carnd_p3/data/IMG/'+img[2].split('/')[-1]
    right_image = cv2.imread(r_name)
    # convert to RGB
    right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('examples/right_image.jpg', right_image_rgb)
    cv2.imwrite('examples/right_image_flip.jpg', cv2.flip(right_image_rgb, 1))
    exit()

