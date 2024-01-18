import os
import shutil

CUSTOM_COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "truck"
}


def get_names_and_images(src_dir):
    all_file_names = []
    all_file_images = []
    # check is directory then open directory
    for file in os.listdir(src_dir):
        if file.endswith('.txt'):
            temp_file_name = f"{src_dir}/{file}"
            all_file_names.append(temp_file_name)
        else:
            image_dir = f"{src_dir}/{file}"
            if os.path.isdir(image_dir):
                all_file_images = [image for image in os.listdir(image_dir)]

    return all_file_names, all_file_images


def get_class_from_file(_file):
    tempFile = _file.split('/')[-1]
    tempFile = tempFile.split('.')[0]
    tempFile = tempFile.split('_')
    tempFile = tempFile[0]

    for clsIdx, clsName in CUSTOM_COCO_CLASSES.items():
        # print(clsIdx, clsName)
        if tempFile == clsName:
            return clsIdx, clsName


def get_image_name(imagelist):
    tempImageList = [image.lstrip('0') for image in imagelist]
    return tempImageList


if __name__ == '__main__':

    # /home/sudiptoshahin/Documents
    # src_dir_path = input("Source directory: ")
    src_dir_path = '/home/sudiptoshahin/Documents/train-data'
    # /home/sudiptoshahin/Documents/
    # des_dir_path = input('Destination directory: ')
    des_dir_path = '/home/sudiptoshahin/Documents'

    # get source directory
    # check desired directory is in or not
    # from *.txt files split the desired files
    # create directory in destination
    src_image_dir = os.path.join(src_dir_path, 'train2017')
    des_images_dir = os.path.join(des_dir_path, 'dataset', 'images')
    des_annotations_dir = os.path.join(des_dir_path, 'dataset', 'annotations')

    if not os.path.exists(des_images_dir):
        os.makedirs(des_images_dir, exist_ok=True)
    if not os.path.exists(des_annotations_dir):
        os.makedirs(des_annotations_dir, exist_ok=True)

    # get the all file name and all images
    file_names, file_images = get_names_and_images(src_dir_path)
    image_file_names = get_image_name(file_images)

    for idx, file_name in enumerate(file_names):
        # test for 1 file only
        # if idx > 0:
        #     break

        class_index, class_name = get_class_from_file(file_name)
        print(idx, class_index, class_name)

        with open(file_name, 'r') as fileTxt:
            tempTexts = fileTxt.read()
            tempTexts = tempTexts.split('\n')
            for line in tempTexts:
                tempLine = line.split(' ')
                if len(tempLine) == 1:
                    continue

                dataFileName = tempLine[0]
                dataImageFileName = f"{dataFileName}.jpg"
                dataTextFileName = f"{dataFileName}.txt"

                # replace image file name with class name
                tempLine[0] = class_index
                line = " ".join(str(txt) for txt in tempLine)
                line = line + '\n'

                temp_des_file = os.path.join(des_annotations_dir, dataTextFileName)
                with open(temp_des_file, 'a') as file:
                    file.write(line)
                    file.close()

                # file_images
                for image_file in image_file_names:
                    # '000000000139.jpg'
                    if dataImageFileName == image_file:
                        temp_src_img_name = dataImageFileName.zfill(16)
                        temp_src_file = os.path.join(src_image_dir, temp_src_img_name)

                        temp_des_img_file = os.path.join(des_images_dir, dataImageFileName)
                        shutil.copy(temp_src_file, temp_des_img_file)


        if idx == len(file_names):
            print('****** completed *******')