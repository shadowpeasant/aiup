import tensorflow as tf
import glob
import numpy as np
from PIL import Image
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

def prepare_dataset(file_set, img_height=100, img_width=100, batch_size=16, shuffle=False):
    
    train_files = sorted(glob.glob(file_set))

    # data are in the shape of #num_samples, heigh, width, #channels)
    a = np.zeros((len(train_files),img_height, img_width,1)).astype(np.float32)

    # store the data in the array in the memory
    for idx, filename in enumerate(train_files):
        #print(f'opening {filename}')
        im = Image.open(filename)
        im = im.resize((img_width, img_height))
        # scale the image values to between 0 and 1
        #print(f'finished {filename}')
        a[idx,:,:,0] = np.array(im)/255.0
        
    # create a dataset with x and its label y, in this case (a,a) i.e. label is same as input
    if shuffle:
        dataset  = tf.data.Dataset.from_tensor_slices((a, a)). \
                        shuffle(1000,seed=42,reshuffle_each_iteration=False). \
                        batch(batch_size)
    else:
        dataset  = tf.data.Dataset.from_tensor_slices((a, a)).batch(batch_size)
    return dataset

dataset = 'UCSDped1'

# setup all the relative path
root_path = os.path.join('UCSD_Anomaly_Dataset.v1p2', dataset)
train_dir = os.path.join(root_path, 'Train')
test_dir = os.path.join(root_path, 'Test')
IMG_HEIGHT=100
IMG_WIDTH=100
BATCH_SIZE=16

if __name__ == '__main__':
    test_subdir = os.path.join(train_dir, '*/*.tif')
    train_dataset = prepare_dataset(test_subdir,
                                    img_height=IMG_HEIGHT, 
                                    img_width=IMG_WIDTH, 
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)