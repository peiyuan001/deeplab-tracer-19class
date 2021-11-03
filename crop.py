import copy
import os
from PIL import Image
import numpy as np
import random


def random_int_list(start, stop, length, seed=0):
    random.seed(seed)
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

def random_beta_int_list(start, stop, length, alpha=2, beta=2, seed=0):
    np.random.seed(seed)
    random_list = []
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    ran= stop-start
    size=[int(length)]
    temp_list =np.random.beta(alpha, beta, size)
    for index in range(length):
        random_list.append(int(start + ran * temp_list[index]))
    return random_list

def cropping(img_path, img_crop_path, img_patch_path, datasize, mode='normal', sd = 0, a=2, b=2): #sd:seed
    Bat = int(datasize / 4)
    imgList = os.listdir(img_path)  # 得到1文件下所指文件内的图片
    print('img list:', imgList)
    dir_test = os.path.join(img_path, imgList[1])
    image_test = Image.open(dir_test).convert('RGB')
    w = image_test.size[0]  # 获取图片宽度
    h = image_test.size[1]  # 获取图片高度
    # pick img randomly
    if mode == 'normal':
        w_c = random_int_list(1, w-1, Bat, seed=0)
        h_c = random_int_list(1, h - 1, Bat, seed=0)
        r_Index = random_int_list(1, len(imgList), len(imgList), seed=sd)
    elif mode == 'beta':
        w_c = random_beta_int_list(1, w - 1, Bat, alpha=a, beta=b, seed=0)
        h_c = random_beta_int_list(1, h - 1, Bat, alpha=a, beta=b, seed=0)
        r_Index = random_beta_int_list(1, len(imgList), len(imgList), alpha=a, beta=b, seed=sd)
    else:
        print('Unknown node')
        return

    for each_batch in range(Bat):  # 遍历，进行批量转
        # Read img
        print('Processing batch:', each_batch, ': image', each_batch * 4, '-', each_batch * 4 + 3)
        dir_1 = os.path.join(img_path, imgList[r_Index[each_batch * 4]])
        dir_2 = os.path.join(img_path, imgList[r_Index[each_batch * 4 + 1]])
        dir_3 = os.path.join(img_path, imgList[r_Index[each_batch * 4 + 2]])
        dir_4 = os.path.join(img_path, imgList[r_Index[each_batch * 4 + 3]])
        # img = image.convert('RGB')
        image_1r = Image.open(dir_1).convert('RGB')
        image_2r = Image.open(dir_2).convert('RGB')
        image_3r = Image.open(dir_3).convert('RGB')
        image_4r = Image.open(dir_4).convert('RGB')

        img_1 = image_1r.crop([0, 0, w_c[each_batch], h_c[each_batch]])  # 获取左上1/4的图片
        #img_1.save(os.path.join(img_crop_path, imgList[r_Index[each_batch * 4]]))

        img_2 = image_2r.crop([w_c[each_batch], 0, w, h_c[each_batch]])  # 获得右上1/4的图片
        #img_2.save(os.path.join(img_crop_path, imgList[r_Index[each_batch * 4 + 1]]))

        img_3 = image_3r.crop([0, h_c[each_batch], w_c[each_batch], h])  # 获取左下1/4的图片
        #img_3.save(os.path.join(img_crop_path, imgList[r_Index[each_batch * 4 + 2]]))

        img_4 = image_4r.crop([w_c[each_batch], h_c[each_batch], w, h])  # 获取右下1/4的图片
        #img_4.save(os.path.join(img_crop_path, imgList[r_Index[each_batch * 4 + 3]]))

        # 4个图像拼接成一个
        target = Image.new('RGB', (w, h))  # 首先创建一块背景布
        for i in range(0, 4):  # 循环4次拼接图片
            if i == 0:
                location = (0, 0)  # 放置位置在左上
                target.paste(img_1, location)
            if i == 1:
                location = (w_c[each_batch], 0)  # 放置位置在右上
                target.paste(img_2, location)
            if i == 2:
                location = (0, h_c[each_batch])  # 放置位置在左下
                target.paste(img_3, location)
            if i == 3:
                location = (w_c[each_batch], h_c[each_batch])  # 放置位置在右下
                target.paste(img_4, location)

        target.save(os.path.join(img_patch_path, imgList[each_batch]))


if __name__ == '__main__':
    img_path = os.path.join(os.getcwd(), 'datasets', 'GTA5', 'images')
    img_crop_path = os.path.join(os.getcwd(), 'datasets', 'GTA5', 'crop_images')
    img_patch_path = os.path.join(os.getcwd(), 'datasets', 'GTA5', 'patch_images')
    label_path = os.path.join(os.getcwd(), 'datasets', 'GTA5', 'labels')
    label_crop_path = os.path.join(os.getcwd(), 'datasets', 'GTA5', 'crop_labels')
    label_patch_path = os.path.join(os.getcwd(), 'datasets', 'GTA5', 'patch_labels')
    alpha = 2
    beta = 2
    seed = 0
    # cropping(img_path, img_crop_path, img_patch_path, 5000, sd=seed)
    # cropping(label_path, label_crop_path, label_patch_path, 5000, sd=seed)
    cropping(img_path, img_crop_path, img_patch_path, 5000, mode='beta', sd=seed, a=alpha, b=beta)
    cropping(label_path, label_crop_path, label_patch_path, 5000, mode='beta', sd=seed, a=alpha, b=beta)
