import time
import torch
import numpy as np

from skimage.segmentation import felzenszwalb


def mask_images(images, params):
    # Get segmentation parameters
    sigma = params['sigma']
    scale = params['scale']
    min_size = params['min_size']

    masked_imgs, orig_imgs = [], []
    for img in images:
        # Reformat input image
        new_img = np.squeeze(img * 255).astype(np.uint8)
        new_img = np.swapaxes(np.swapaxes(new_img, 0, 1), 1, 2)

        # Perform image segmentation
        segments = segment(new_img, sigma, scale, min_size)
        all_pixels = segment_pixels(segments)

        # Create a mask tensor for each segment
        for pixels in all_pixels:
            masked_img = np.copy(img)[np.newaxis,:,:,:]
            masked_img[:,:, pixels[0, :], pixels[1, :]] = 1
            masked_imgs.append(masked_img)
            orig_imgs.append(np.copy(img)[np.newaxis,:,:,:])
    
    return np.vstack(masked_imgs), np.vstack(orig_imgs)

def segment(img, sigma, scale, min_size):
    return felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
   
def segment_pixels(segments):
    all_pixels = []
    for segment in np.unique(segments):
        # Get pixels coorresponding to current segment
        pixels = np.array(np.where(segments == segment))
        all_pixels.append(torch.LongTensor(pixels))
    return all_pixels

def segment_img(input, segments):
    colors = []
    for segment in np.unique(segments):
        colors.append(np.random.randint(0, 255, size=(3,), dtype=int))
    
    output = np.zeros_like(input)
    height, width, _ = np.shape(input)
    for i in range(height):
        for j in range(width):   
            output[i, j, :] = colors[segments[i, j]]
    return output

def main():
    import os
    import argparse
    from PIL import Image
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--img_folder', type=str, default='data/')
    parser.add_argument('--output_dir', type=str, default='images/')
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--sigma', type=float, default=0.8)
    parser.add_argument('--scale', type=int, default=750)
    parser.add_argument('--min_size', type=int, default=50)
    args = parser.parse_args()

    # Create directory to store segmented images
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    _, cluster_names, _ = list(os.walk(args.img_folder))[0]
    for idx, cluster_name in enumerate(cluster_names):
        cluster_dir = os.path.join(args.img_folder, cluster_name)
        for file in os.listdir(cluster_dir):
            if not (file.endswith('.jpg') or file.endswith('.png')):
                continue

            # Read input image
            img = Image.open(os.path.join(cluster_dir, file))
            height = args.height if args.height is not None else img.size[1] 
            width  = args.width  if args.width  is not None else img.size[0] 
            img = img.resize((width, height))
            img = np.array(img)

            # Segment image
            segments = segment(img, args.sigma, args.scale, args.min_size)
            output = segment_img(img, segments)

            # Plot segmented image
            plt.subplot(1, 2, 1)
            im = plt.imshow(img)
            plt.subplot(1, 2, 2)
            im = plt.imshow(output)
            plt.savefig(os.path.join(args.output_dir, file))

if __name__=="__main__":
    main()