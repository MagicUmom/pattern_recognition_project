import os
from PIL import Image

img_true_dir = "dataset/true/"
img_false_dir = "dataset/false/"

T_target_path = "dataset/true_resize_rotate/"
F_target_path = "dataset/false_resize_rotate/"

for image in os.listdir(img_true_dir):
    im = Image.open(img_true_dir+image)
    # img_name = image
    img_name = image.split('.')[0]
    # print(type(img_name))
    im2 = im.resize((200, 200), Image.BILINEAR)
    im2.save(T_target_path+img_name + '_0.png','PNG')
    im2.rotate(90, Image.BILINEAR).save(T_target_path + img_name + '_1.png','PNG')
    im2.rotate(180,Image.BILINEAR).save(T_target_path + img_name + '_2.png','PNG')
    im2.rotate(270,Image.BILINEAR).save(T_target_path + img_name + '_3.png','PNG')

# for image in os.listdir(img_false_dir):
#     im = Image.open(img_false_dir+image)
#     # img_name = image
#     img_name = image.split('.')[0]
#     # print(type(img_name))
#     im2 = im.resize((200, 200), Image.BILINEAR)
#     im2.save(F_target_path+img_name + '_0.png','PNG')
#     im2.rotate(90, Image.BILINEAR).save(F_target_path + img_name + '_1.png','PNG')
#     im2.rotate(180,Image.BILINEAR).save(F_target_path + img_name + '_2.png','PNG')
#     im2.rotate(270,Image.BILINEAR).save(F_target_path + img_name + '_3.png','PNG')
