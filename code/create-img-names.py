import os
from os import listdir

# please use relative path
path = r"/Users/yooniiechi/Desktop/AI/AI_PROJECT-ASL/gesture/train/A"

files = os.listdir(path)
i = 0

for index, file in enumerate(files):
    
    # current_file = os.path.join(path, file)
    # print(current_file)
    # new_name = ["A - " + str(index), '.jpg']
    # print(index)
                             
    # be sure to change the Letter each time
    os.rename(os.path.join(path, file), os.path.join(path, ''.join(["A-" + str(index), '.jpg'])))

    i = i+1
    
# for (count, images) in enumerate(os.listdir(path)):
    # dst = "A" + "-" + str(num)
    # print(dst)
    
    # path_to_img = path + images
    # print(path_to_img)
    ß
