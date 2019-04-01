import re
import glob
import numpy as np
import argparse
import imageio as io
import os


def get_indices_from_dir(dir):
    """
    take all files like: image_000000.png and return an array of int64s
    :param dir: path of image directors
    :return: np array
    """
    regex = re.compile(r'\d+')
    list_of_indices = []
    for filename in glob.glob(dir + '/*'):
        list_of_indices.append([int(x) for x in regex.findall(filename)][0])

    return np.array(list_of_indices, dtype=np.int)


def make_blank_modal_segmasks(image_dir, output_dirs=['modal_segmasks', 'segmasks_filled'], replace_path='color_ims'):
    """

    :param image_dir: dir of the images
    :param output_dir: dir for the segmasks
    :return: nothing
    """
    for image_filename in glob.glob(image_dir + '/*'):
        image = io.imread(image_filename)
        image_shape = image.shape[:2]
        blank_image = np.zeros(list(image_shape))
        for output_dir in output_dirs:
            output_image_filename = image_filename.replace(replace_path, output_dir)
            io.imsave(output_image_filename, blank_image)


def convert_depth_images(input_directory, output_directory):
    for i, image_filename in enumerate(sorted(glob.glob(input_directory + '/*.png'))):
        image_array = io.imread(image_filename)
        new_image_array = np.copy(image_array[:, :, :2])
        depth_array = np.ndarray(shape=image_array.shape[:2], dtype='<H', buffer=memoryview(new_image_array))
        output_image_filename = os.path.join(output_directory, 'image_{:06d}.png'.format(i))
        io.imsave(output_image_filename, depth_array, format='png')


if __name__ == '__main__':
    conf_parser = argparse.ArgumentParser(description="make indices file")
    conf_parser.add_argument("--indices_dir",
                             action="store",
                             default="./",
                             dest="indices_dir",
                             type=str,
                             help="path to the configuration file")
    conf_parser.add_argument("--output_indices_file",
                             action="store",
                             default="./",
                             dest="output_indices_file",
                             type=str,
                             help="output_indices_file")

    conf_parser.add_argument("--input_depth_dir",
                             action="store",
                             default="./",
                             dest="input_depth_dir",
                             type=str,
                             help="input_depth_directory")

    conf_parser.add_argument("--output_depth_dir",
                             action="store",
                             default="./",
                             dest="output_depth_dir",
                             type=str,
                             help="output_depth_directory")


    conf_args = conf_parser.parse_args()
    if conf_args.input_depth_dir != './':
        convert_depth_images(conf_args.input_depth_dir, conf_args.output_depth_dir)

    else:
        indices = get_indices_from_dir(conf_args.indices_dir)
        np.save(conf_args.output_indices_file, indices)
        make_blank_modal_segmasks(conf_args.indices_dir)
