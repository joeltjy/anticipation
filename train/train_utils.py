import os
import sys
from glob import glob

import shutil

from tqdm import tqdm
sys.path.append("..")
from anticipation.config import *
print(sys.path)
def split_all_compound_files(dir, match_string):
    """
    Places the compound files in dir into folders defined by MAESTRO_SPLITS.
    In this case, the compound_files will be sorted into 0, 1, 2, ..., e, f
    so that the general tokenize_lakh preprocessing pipelien can be run on it.
    """
    print("inputs", os.path.join(dir, match_string))

    if not os.path.exists(dir):
        print(f"Error: Directory {dir} does not exist.")
        return

    files = glob(os.path.join(dir, match_string))
    print("number of files", len(files))
    for i in MAESTRO_SPLITS:
        if not os.path.exists(os.path.join(dir, i)):
            os.mkdir(os.path.join(dir, i))
    
    n = len(MAESTRO_SPLITS) # in our case, n = 16.
    for j, file in enumerate(files):
        move_file(file, os.path.join(dir, MAESTRO_SPLITS[j % n]))


    print("Success!")
    
def move_file(src, dst):
    try:
        if os.path.exists(dst):
            print("Warning: File already exists in the destination!")
        shutil.move(src, dst)
        print("File moved successfully!")
    except FileNotFoundError:
        print("Error: Source file not found!")
    except PermissionError:
        print("Error: Permission denied!")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    import sys
    method_name = sys.argv[1]
    dir = sys.argv[2]
    match_string = sys.argv[3]
    split_all_compound_files(dir, match_string)