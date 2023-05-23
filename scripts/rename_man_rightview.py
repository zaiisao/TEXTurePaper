import os

folder = "/home/jahn/dir_dataset/man/man_right"
for filename in os.listdir(folder):
    src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{folder}/{filename.replace('man_leftview', 'man_rightview')}"
        
    # rename() function will
    # rename all the files
    os.rename(src, dst)