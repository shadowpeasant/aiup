from PIL import Image
import glob
import matplotlib.pyplot as plt
import math
import os
import zipfile
import tensorflow as tf
import numpy as np
from scipy import signal
from os import system
import urllib.request

from tqdm import tqdm


try:
    import wget
    print('\nWget Module was installed')
except ImportError:
    system("pip install wget")
    import wget
    
    
root_logdir = os.path.join(os.curdir, "tb_logs")

def fix_cudnn_bug(): 
    # during training, tf will throw cudnn initialization error: failed to get convolution algos
    # the following codes somehow fix it
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

def plot_training_loss(history):
    train_loss = history['loss']
    if 'val_loss' in history:
        val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'bo', label='Traintrain_lossing loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()
    
def create_gif(image_folder, output_file, img_type='png',):
    # Create the frames
    frames = []

    # files need to be sorted from 1...n so that the video is played in correct sequence
    imgs = sorted(glob.glob(f'{image_folder}/*.{img_type}'))
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    
    # Save into a GIF file that loops forever
    frames[0].save(output_file, format='gif',
                append_images=frames[1:],
                save_all=True,
                duration=120, loop=0)

def plot_image(image):
    '''if image is a file, then open the file first'''
    
    if type(image) == str:
        img = Image.open(image)
    elif type(image) == tf.python.framework.ops.EagerTensor:
        if len(image.shape) == 4:  # the tensor with batch axis
            img =  image[0][:,:,0]
        else:
            img = image[:,:,0]

    plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis("off")
    
    
def display_images(image_folder, image_range=(1,10), max_per_row=5):
    start, end = image_range
    num_images = end - start
    images = []
    for i in range(start, end):
        images.append(os.path.join(image_folder, '{:03d}.tif'.format(i)))

    nrows = math.ceil(num_images/max_per_row)
    fig = plt.figure(figsize=(max_per_row * 3, nrows * 2))
    for index, image in enumerate(images):
        plt.subplot(nrows, max_per_row, 1 + index)
        plot_image(image)
    #fig.save('fig.png')
    plt.show()

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_data(data_dir, url, extract=True, force=False):
    # if not force download and directory already exists
    if not force and os.path.exists(data_dir):
        print('dataset directory already exists, skip download')
        return
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        filename, _ = urllib.request.urlretrieve(url, reporthook=t.update_to)
        if extract: 
            with zipfile.ZipFile(filename, 'r') as zip:
                zip.extractall(data_dir)
        os.remove(filename)

def show_reconstructions(model, image):
    im = Image.open(image)
    im = np.array(im.resize((100,100)))/255.
    im = np.expand_dims(np.expand_dims(im, axis=0), axis=3)
    reconstructed = model.predict(im)
    plt.subplot(1, 2, 1)
    plt.imshow(im[0,:,:,0], cmap=plt.cm.gray, interpolation='nearest')
    #plt.xlabel('original')
    plt.title('original')
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed[0,:,:,0], cmap=plt.cm.gray, interpolation='nearest')
    #plt.xlabel('reconstructed')
    plt.title('reconstructed')
    plt.axis("off")
    
def plot_reconstruction_loss(img, losses, counter):
    if not os.path.exists('losses'):
        os.mkdir('losses')
    plt.ioff()
    fig = plt.figure(figsize=(6, 2))
    #plt.yticks(np.arange(0, 0.03, step=0.005))
    x = np.arange(1,201)
    y = np.zeros(200)
    # show original image
    fig.add_subplot(121)
    plt.title(f'frame {counter}')
    plt.set_cmap('gray')
    plt.imshow(img)

    fig.add_subplot(122)
    #plt.yticks(np.arange(0, 0.015, step=0.005))
    plt.ylim(0,0.015)
    plt.title('reconstruction loss')
    plt.plot(x,y)
    plt.plot(losses)

    #plt.show() 
    
    fig.savefig('losses/{:0>3d}.png'.format(counter))
    plt.ion()
    plt.close()
  
    
def create_losses_animation(model, dataset, gif_file):
    mse = tf.keras.losses.MeanSquaredError()
    losses = []
    counter = 0
    for image, _  in dataset:
        counter = counter + 1
        output = model.predict(image)
        loss = mse(image, output)
        losses.append(loss)
        plot_reconstruction_loss(image[0,:,:,0], losses, counter)
    
    create_gif('losses', gif_file)
    
def plot_comparisons(img, output, diff, H, threshold, counter):
    if not os.path.exists('images'):
        os.mkdir('images')
    plt.ioff()
    #print('inside plot, imgshape {}'.format(img.shape))
    fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(10, 5))
    ax0.set_axis_off()
    ax1.set_axis_off()
    ax2.set_axis_off()
    
    ax0.set_title('input image')
    ax1.set_title('reconstructed image')
    ax2.set_title('diff ')
    ax3.set_title('anomalies')
    #ax4.set_title('H')
    ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest') 
    ax1.imshow(output, cmap=plt.cm.gray, interpolation='nearest')   
    ax2.imshow(diff, cmap=plt.cm.viridis, vmin=0, vmax=255, interpolation='nearest')  
    ax3.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    #ax4.imshow(H, cmap=plt.cm.gray, interpolation='nearest')
    
    x,y = np.where(H > threshold)
    ax3.scatter(y,x,color='red',s=0.1) 

    plt.axis('off')
    
    fig.savefig('images/{:0>3d}.png'.format(counter))
    plt.close()
    plt.ion()
    
def identify_anomaly(model, dataset, gif_file, threshold=4):
    threshold = threshold*255
    counter = 0;
    for image, _  in dataset:
        counter = counter + 1
        output = model.predict(image)
        output = tf.multiply(output,255.)
        img = tf.multiply(tf.cast(image, tf.float32), 255.)
        diff = tf.abs(tf.subtract(output,img))
        tmp = diff[0,:,:,0]
        #print(tmp)
        H = signal.convolve2d(tmp, np.ones((4,4)), mode='same')
        #print(H)
        plot_comparisons(img[0,:,:,0], output[0,:,:,0], diff[0,:,:,0], H, threshold, counter)

    create_gif('images', gif_file)
    



class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, reporthook=t.update_to)
        
        
if __name__ == '__main__':
    
    image_folder = r'C:\Users\kheng\.keras\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test024'
    #create_gif(image_folder, 'mygif.gif', img_type='tif')

    # images = []
    # for i in range(10):
    #     images.append(os.path.join(image_folder, '{:03d}.tif'.format(i+1)))
    
    # print(images)
    image_folder = os.path.join(dataset_root_dir, 'UCSDped1', 'Train', 'Train001')
    display_images(image_folder,image_range=(1,6), max_per_row=5)

    # url = 'https://sdaaidata.s3-ap-southeast-1.amazonaws.com/UCSD_Anomaly_Dataset.v1p2.zip'

    # download(url, extract=True)