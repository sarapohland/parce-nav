import os
import argparse
from moviepy.editor import ImageSequenceClip

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('img_folder', type=str)
    parser.add_argument('--video_file', type=str, default='results/visualization/video.mp4')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    # Create folder to save video file if it doesn't exist
    folder = os.path.dirname(args.video_file)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(args.img_folder) if os.path.isfile(os.path.join(args.img_folder, f))]

    # Sort the image files based on their names (assuming names correspond to time)
    image_files.sort(key=lambda x: float(x.split('.png')[0]))

    # Create a list of image paths
    image_paths = [os.path.join(args.img_folder, filename) for filename in image_files]

    # Load images and create video
    clip = ImageSequenceClip(image_paths, fps=args.fps)
    clip.write_videofile(args.video_file, codec='libx264')

if __name__ == "__main__":
    main()