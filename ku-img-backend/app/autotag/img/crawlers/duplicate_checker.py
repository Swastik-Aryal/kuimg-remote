import os
import imagehash
from PIL import Image
from collections import defaultdict


class Duplicates:
    def __init__(self):
        pass

    @staticmethod 
    def find_duplicate_images(folder):
        # Dictionary to store image hashes
        image_hashes = defaultdict(list)
        
        # Calculate the hash for each image and store it in the dictionary
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(folder, filename)
                try:
                    with Image.open(image_path) as img:
                        img_hash = imagehash.phash(img)
                        image_hashes[img_hash].append(filename)
                except Exception as e:
                    print(f"Could not process image {filename}: {e}")
        
        # Identify duplicates
        duplicates = {hash_value: files for hash_value, files in image_hashes.items() if len(files) > 1}
        
        return duplicates

    @staticmethod
    def remove_duplicates(folder):
        # Keep one copy of each set of duplicates and delete the rest
        
        duplicates = Duplicates.find_duplicate_images(folder)
        for files in duplicates.values():
            for file in files[1:]:  # Keep the first file, remove the rest
                os.remove(os.path.join(folder, file))
        
        if duplicates:
            print("Found and fixed the following duplications:")
            for hash_value, files in duplicates.items():
                print(f"Hash: {hash_value}")
                for file in files:
                    print(f"  {file}")
            
            print('After fixing. No of images = {}'.format(len(os.listdir(folder))))
        


if __name__ == "__main__":
    # Specify the directory containing the images
    image_folder = r"data\images\cat"


    # Find and remove duplicates
    duplicates = Duplicates.find_duplicate_images(image_folder)
    print(duplicates)
    Duplicates.remove_duplicates(image_folder)

