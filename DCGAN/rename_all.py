import os


directory = "/home/mahmoud-sayed/Desktop/Code/Python/Models From Scratch/DCGAN/images"
files = os.listdir(directory)


for i,filename in enumerate(files):
    old_path = os.path.join(directory, filename)
    new_filename = f'{i}.jpg'.ljust(3,'0')
    new_path = os.path.join(directory, new_filename)
    os.rename(old_path, new_path)