import albumentations as A
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import codecs
import json
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

KEYPOINT_COLOR = (0, 255, 0)  # Green

def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=15):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)


def data_aug():
    # Declare an augmentation pipeline
    transform = A.Compose([
#        A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=0.5),
#        A.VerticalFlip(p=0.5),
#        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']), keypoint_params=A.KeypointParams(format='xy'))

    # 3-
    #image = cv2.imread("images/labelme/1639726903.jpg")
    image = cv2.imread("images/labelme/1639726322.jpg")
    #cv2.imwrite("./sss1.jpg", image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # bboxes = [
    #     [0, 117, 638, 247],
    # ]
    # keypoints = [
    #     (638,119),
    #     (0, 253),
    #     (1, 157),
    #     (637, 124),
    #     (636, 238),
    #     (2, 250),
    # ]

    bboxes = [
        [0, 208, 639, 314],
    ]
    keypoints = [
        (0, 208),
        (639, 314),
        (0, 214),
        (637, 207),
        (639, 307),
        (0, 314),
    ]
    image1 = image.copy()
    cv2.rectangle(image, (0, 208),(639, 314), (0, 255, 0))
    #cv2.circle(image, (0, 208), 5, (0, 255, 0), -1)
    #cv2.circle(image, (639, 314), 5, (0, 255, 0), -1)
    cv2.circle(image, (0, 214), 5, (0, 255, 255), -1)
    cv2.circle(image, (637, 207), 5, (0, 255, 255), -1)
    cv2.circle(image, (639, 307), 5, (0, 255, 255), -1)
    cv2.circle(image, (0, 314), 5, (0, 255, 255), -1)
    cv2.imwrite("./sss1.jpg", image)

    class_labels = ['false']
    #class_categories = ['animal', 'animal', 'item']
    # 4-
    transformed = transform(image=image1, bboxes=bboxes, class_labels=class_labels, keypoints=keypoints)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_keypoints = transformed['keypoints']
    transformed_class_labels = transformed['class_labels']
    diameter = 3

    cv2.rectangle(transformed_image, (int(transformed_keypoints[0][0]), int(transformed_keypoints[0][1])), (int(transformed_keypoints[1][0]), int(transformed_keypoints[1][1])), (0, 255, 0))
    for (x, y) in transformed_keypoints:
        print(int(x))
        print(int(y))
        cv2.circle(transformed_image, (int(x), int(y)), diameter, (0, 255, 0), -1)
    #vis_keypoints(transformed_image, transformed_keypoints)
    cv2.imwrite("./sss.jpg", transformed_image)


def data_aug_data(p_0, p_1, pic, rename):
    # Declare an augmentation pipeline

    if rename == '_horizon':
        transform = A.Compose([
            #        A.RandomCrop(width=450, height=450),
                    A.HorizontalFlip(p=1.0),
            #        A.VerticalFlip(p=0.5),
            #A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
            keypoint_params=A.KeypointParams(format='xy'))

    if rename == '_Blur':
        transform = A.Compose([
    #        A.RandomCrop(width=450, height=450),
    #        A.HorizontalFlip(p=0.5),
    #        A.VerticalFlip(p=0.5),
    #        A.RandomBrightnessContrast(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.5),  # 使用随机大小的内核将运动模糊应用于输入图像。
                A.MedianBlur(blur_limit=3, p=0.5),  # 中值滤波
                A.Blur(blur_limit=3, p=0.5),  # 使用随机大小的内核模糊输入图像。
            ], p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']), keypoint_params=A.KeypointParams(format='xy'))


    # 3-
    image = cv2.imread(pic)
    #cv2.imwrite("./sss1.jpg", image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = [
        [min(p_0[:, 0]), min(p_0[:, 1]), max(p_0[:, 0]), max(p_0[:, 1])],
    ]

    keypoints = [
        (min(p_0[:, 0]), min(p_0[:, 1])),
        (max(p_0[:, 0]), max(p_0[:, 1])),
        (p_1[0, 0],p_1[0, 1]),
        (p_1[1, 0], p_1[1, 1]),
        (p_1[2, 0], p_1[2, 1]),
        (p_1[3, 0], p_1[3, 1]),
    ]

    class_labels = ['false']
    #class_categories = ['animal', 'animal', 'item']
    # 4-
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels, keypoints=keypoints)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_keypoints = transformed['keypoints']
    transformed_class_labels = transformed['class_labels']
    return transformed_keypoints, transformed_image

def data_aug_pic( pic, rename):
    # Declare an augmentation pipeline

    if rename == '_horizon':
        transform = A.Compose([
            #        A.RandomCrop(width=450, height=450),
                    A.HorizontalFlip(p=1.0),
            #        A.VerticalFlip(p=0.5),
            #A.RandomBrightnessContrast(p=0.2),
        ])

    if rename == '_Blur':
        transform = A.Compose([
    #        A.RandomCrop(width=450, height=450),
    #        A.HorizontalFlip(p=0.5),
    #        A.VerticalFlip(p=0.5),
    #        A.RandomBrightnessContrast(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.5),  # 使用随机大小的内核将运动模糊应用于输入图像。
                A.MedianBlur(blur_limit=3, p=0.5),  # 中值滤波
                A.Blur(blur_limit=3, p=0.5),  # 使用随机大小的内核模糊输入图像。
            ], p=1.0),
        ])


    # 3-
    image = cv2.imread(pic)
    #cv2.imwrite("./sss1.jpg", image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    #class_categories = ['animal', 'animal', 'item']
    # 4-
    transformed = transform(image=image)
    transformed_image = transformed['image']

    return transformed_image


def labelme_json_voc_widerface(rename):
    # 1.标签路径
    labelme_path = ".\\labelme_to_voc_widerface\\labelme\\"  # 原始labelme标注数据路径
    saved_path = ".\\labelme_to_voc_widerface\\VOC2007\\"  # 保存路径
    #labelme_path = ".\\images\\labelme\\"  # 原始labelme标注数据路径
    #saved_path = ".\\images\\VOC2007\\"  # 保存路径

    # 2.创建要求文件夹
    if not os.path.exists(saved_path + "Annotations"):
        os.makedirs(saved_path + "Annotations")
    if not os.path.exists(saved_path + "JPEGImages/"):
        os.makedirs(saved_path + "JPEGImages/")
    if not os.path.exists(saved_path + "ImageSets/Main/"):
        os.makedirs(saved_path + "ImageSets/Main/")

    # 3.获取待处理文件
    files = glob(labelme_path + "*.json")
    files = [i.split("\\")[-1].split(".json")[0] for i in files]

    #file_handle = open(".\\images\\save_widerface_result_horizon1.txt", mode='w')
    if rename == '_horizon':
        file_handle = open("labelme_to_voc_widerface/save_widerface_result_horizon.txt", mode='w')
    if rename == '_Blur':
        file_handle = open("labelme_to_voc_widerface/save_widerface_result_blur.txt", mode='w')

    # 4.读取标注信息并写入 xml
    for json_file_ in files:
        json_filename = labelme_path + json_file_ + ".json"
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        height, width, channels = cv2.imread(labelme_path + json_file_ + ".jpg").shape
        with codecs.open(saved_path + "Annotations/" + json_file_ + rename+".xml", "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'STOP_LINE' + '</folder>\n')
            xml.write('\t<filename>' + json_file_ +rename+ ".jpg" + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>The STOP LINE Database</database>\n')
            xml.write('\t\t<annotation>PASCAL VOC</annotation>\n')
            xml.write('\t\t<image>flickr</image>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t</source>\n')
            xml.write('\t<owner>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t\t<name>Line</name>\n')
            xml.write('\t</owner>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(width) + '</width>\n')
            xml.write('\t\t<height>' + str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t<segmented>0</segmented>\n')
            # print(json_file["shapes"][0]["points"])

            points_0 = np.array(json_file["shapes"][0]["points"])
            points_1 = np.array(json_file["shapes"][1]["points"])
            transform_data, new_img = data_aug_data(points_0, points_1, labelme_path + json_file_ + ".jpg",rename)

            cv2.imwrite(saved_path + "JPEGImages\\"+json_file_+ rename+".jpg", new_img)

            xmin = min(transform_data[0][0],transform_data[1][0])
            xmax = max(transform_data[0][0],transform_data[1][0])
            ymin = min(transform_data[0][1],transform_data[1][1])
            ymax = max(transform_data[0][1],transform_data[1][1])
            label = json_file["shapes"][0]["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + str(label) + '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')


            label_1 = json_file["shapes"][1]["label"]
            x1 = transform_data[2][0]
            y1 = transform_data[2][1]
            x2 = transform_data[3][0]
            y2 = transform_data[3][1]
            x3 = transform_data[4][0]
            y3 = transform_data[4][1]
            x4 = transform_data[5][0]
            y4 = transform_data[5][1]




            '''
            xml.write('\t<object>\n')
            xml.write('\t\t<name>'+str(label_1)+'</name>\n')
            xml.write('\t\t<pose>Unspecified</pose>\n')
            xml.write('\t\t<truncated>1</truncated>\n')
            xml.write('\t\t<difficult>0</difficult>\n')
            xml.write('\t\t<point>\n')
            xml.write('\t\t\t<x1>' + str(int(x1)) + '</x1>\n')
            xml.write('\t\t\t<y1>' + str(int(y1)) + '</y1>\n')
            xml.write('\t\t\t<x2>' + str(int(x2)) + '</x2>\n')
            xml.write('\t\t\t<y2>' + str(int(y2)) + '</y2>\n')
            xml.write('\t\t\t<x3>' + str(int(x3)) + '</x3>\n')
            xml.write('\t\t\t<y3>' + str(int(y3)) + '</y3>\n')
            xml.write('\t\t\t<x4>' + str(int(x4)) + '</x4>\n')
            xml.write('\t\t\t<y4>' + str(int(y4)) + '</y4>\n')
            xml.write('\t\t</point>\n')
            xml.write('\t</object>\n')
            '''
            # print(json_file_ + ".jpg",xmin,ymin,xmax,ymax,label,x1,y1,x2,y2,x3,y3,x4,y4)
            if rename == '_Blur':
                txt_file = "# " + json_file_ + rename+".jpg" + '\n' + str(int(xmin)) + " " + str(int(ymin)) + " " + str(
                    int(xmax-xmin)) + " " + str(int(ymax-ymin)) + " " + str(int(label)) + " " + str(int(x1)) + " " + str(
                    int(y1)) + " " + str(int(x2)) + " " + str(int(y2)) + " " + str(int(x3)) + " " + str(
                    int(y3)) + " " + str(int(x4)) + " " + str(int(y4)) + " " + str(int(label_1))
            if rename == '_horizon':
                txt_file = "# " + json_file_ + rename + ".jpg" + '\n' + str(int(xmin)) + " " + str(
                    int(ymin)) + " " + str(int(xmax - xmin)) + " " + str(int(ymax - ymin)) + " " + str(int(label)) + " " + str(
                    int(x2)) + " " + str(int(y2)) + " " + str(int(x1)) + " " + str(int(y1)) + " " + str(int(x4)) + " " + str(
                    int(y4)) + " " + str(int(x3)) + " " + str(int(y3)) + " " + str(int(label_1))
            file_handle.write(txt_file + '\n')
            xml.write('</annotation>')

    # 5.复制图片到 VOC2007/JPEGImages/下
    #image_files = glob(labelme_path + "*.jpg")
    #print("copy image files to VOC007/JPEGImages/")
    # for image in image_files:
    #     new_name = os.path.basename(image).split(".")[0]+"_abl.jpg"
    #     shutil.copy(image, saved_path + "JPEGImages\\"+new_name)

    # 6.split files for txt
    txtsavepath = saved_path + "ImageSets\\Main\\"
    ftrainval = open(txtsavepath + '\\trainval.txt', 'w')
    ftest = open(txtsavepath + '\\test.txt', 'w')
    ftrain = open(txtsavepath + '\\train.txt', 'w')
    fval = open(txtsavepath + '\\val.txt', 'w')
    #total_files = glob(".\\images\\VOC2007\\Annotations\\*.xml")
    total_files = glob(".\\labelme_to_voc_widerface\\VOC2007\\Annotations\\*.xml")
    total_files = [i.split("\\")[-1].split(".xml")[0] for i in total_files]
    # test_filepath = ""
    for file in total_files:
        #os.path.basename(file).split(".")[0] + "_abl.jpg"
        ftrainval.write(file + "\n")
    # test
    # for file in os.listdir(test_filepath):
    #    ftest.write(file.split(".jpg")[0] + "\n")
    # split
    train_files, val_files = train_test_split(total_files, test_size=0.15, random_state=42)
    # train
    for file in train_files:
        ftrain.write(file +"\n")
    # val
    for file in val_files:
        fval.write(file + "\n")

    ftrainval.close()
    ftrain.close()
    fval.close()

def labelme_json_voc_widerface_copy():
    # 1.标签路径
    labelme_path = "F:\\stop_line_data\\train\\"  # 原始labelme标注数据路径
    saved_path = "F:\\stop_line_data\\label_1\\"  # 保存路径



    # 3.获取待处理文件
    files = glob(labelme_path + "*.json")
    files = [i.split("\\")[-1].split(".json")[0] for i in files]

    file_handle = open("labelme_to_voc_widerface/save_widerface_result_horizon1.txt", mode='w')
    # file_handle = open("labelme_to_voc_widerface/save_widerface_result_blur1.txt", mode='w')

    # 4.读取标注信息并写入 xml
    for json_file_ in files:
        json_filename = labelme_path + json_file_ + ".json"
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        height, width, channels = cv2.imread(labelme_path + json_file_ + ".jpg").shape
        label = json_file["shapes"][0]["label"]
        if int(label) > 0:
            shutil.copy(labelme_path + json_file_ + ".jpg", saved_path)
            shutil.copy(json_filename, saved_path)


def val_horizon_data():
    #txt_path = ".\\labelme_to_voc_widerface\\save_widerface_result_horizon.txt"
    txt_path = ".\\labelme_to_voc_widerface\\save_widerface_result_blur.txt"
    f = open(txt_path, 'r')
    lines = f.readlines()
    isFirst = True
    imgs_path = []
    words = []
    labels = []
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            if isFirst is True:
                isFirst = False
            else:
                labels_copy = labels.copy()
                words.append(labels_copy)
                #labels.clear()
            path = line[2:]
            #path = txt_path.replace('save_widerface_result_horizon.txt', 'VOC2007\\JPEGImages\\') + path
            path = txt_path.replace('save_widerface_result_blur.txt', 'VOC2007\\JPEGImages\\') + path
            imgs_path.append(path)
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            labels.append(label)

    for index in range(0, len(imgs_path)):
        img_raw = cv2.imread(imgs_path[index])
        #print(img_raw.shape)
        #print(int(labels[index][0]))

        cv2.rectangle(img_raw, (int(labels[index][0]), int(labels[index][1])), (int(labels[index][0])+int(labels[index][2]), int(labels[index][1])+int(labels[index][3])), (0, 0, 255), 2)

        cv2.circle(img_raw, (int(labels[index][5]), int(labels[index][6])), 2, (255, 255, 0), 8)
        cv2.circle(img_raw, (int(labels[index][7]), int(labels[index][8])), 2, (0, 255, 255), 8)
        cv2.circle(img_raw, (int(labels[index][9]), int(labels[index][10])), 2, (255, 0, 255), 8)
        cv2.circle(img_raw, (int(labels[index][11]), int(labels[index][12])), 2, (0, 255, 0), 8)
        #saved_path = os.path.join(os.path.dirname(imgs_path[index]), os.path.basename(imgs_path[index]).split(".")[0]+"_1.jpg")
        saved_path = os.path.join(".\\labelme_to_voc_widerface\\save_result", os.path.basename(imgs_path[index]).split(".")[0]+"_1.jpg")
        cv2.imwrite(saved_path, img_raw)
        #shutil.copy(imgs_path[index], saved_path)



def data_album(dataDir, rename):
    list_dirs = os.walk(dataDir)
    for root, dirs, files in list_dirs:
        # for d in dirs:
        #     print("@@@@@@@@@@@@@@@@@@@@")
        #     print(os.path.join(root,d))
        for f in files:
            #print("********************")
            #./images\VOC2007\ImageSets\Main
            #print(root)
            path_name = root
            #./images\VOC2007\ImageSets\Main\val.txt
            #print(os.path.join(root,f))
            pic_path = os.path.join(root,f)

            new_img = data_aug_pic(pic_path,rename)
            cv2.imwrite(os.path.join(".\\data_album", os.path.basename(pic_path).split(".")[0]+rename+".jpg"), new_img)


#data_aug()
#将label为1的 jpg和json复制在新的文件夹
#labelme_json_voc_widerface_copy()

#验证水平翻转和模糊后关键点和框的正确性，画出来
#val_horizon_data()

#单纯的对图片做模糊和水平翻转，没有坐标值和数据
# data_album(".\\data_album","_horizon")
data_album(".\\data_album","_Blur")

#对labelme标注数据做增强，并将增强的数据转换成voc和widerface格式
#labelme_json_voc_widerface("_horizon")
# labelme_json_voc_widerface("_Blur")




