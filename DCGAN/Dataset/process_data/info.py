import os

path =  '/home/mahmoud-sayed/Desktop/Dataset/oxford flowers/dataset'

train = os.path.join(path, 'train')
test = os.path.join(path, 'test')
val = os.path.join(path, 'valid')


# train_folders = os.listdir(train)
# test_folders = os.listdir(test)
# val_folders = os.listdir(val)
# print(train_folders)
def count_images(path):
    _sum = 0
    _classes = os.listdir(path)
    for _class in _classes:
        full_path = os.path.join(path, _class)
        _sum += len(os.listdir(full_path))

    print(_sum)


count_images(train)
count_images(val)
print(len(os.listdir(test)))