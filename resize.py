import os
from PIL import Image

img_true_dir = "dataset/true_for_train/"
img_false_dir = "dataset/false/"

target_path = "dataset/true_resize_square/"

min_w = 636
min_h = 397

width = 200
height = 200

for image in os.listdir(img_true_dir):
    if not image.startswith('.') :
        im = Image.open(img_true_dir + image)
    # w,h = im.size
    # ratio = float(min_h) / im.size[1]
    # width = int(im.size[0] * ratio)

    # if width < min_w :
    #     print(image,width)
        im2 = im.resize((width, height), Image.BILINEAR)
        im2.save(target_path+image.split('.')[0]+'.png' , 'PNG')
    # print(w/h)