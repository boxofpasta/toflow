import glob
import os
import argparse


def get_triplet_filepaths(input_dir, output_dir):
    """
    :param input_dir: Dataset directory. Each video sequence of images is assumed to be contained within its own folder.
    :param output_dir: The output file path root.
    :return: A list of triplets of (input_a_path, input_c_path, output_b_path).
             The network should predict using images read in from input paths, and output to output_b_path.
             output_b_path will mirror input_b_path, i.e the input_dir substring is replaced with output_dir.
    """
    all_subdirs = get_subdirs(input_dir)

    # Go through each subdirectory and form triplets of (input_a_path, input_c_path, output_b_path).
    # output_b_path will mirror input_b_path, i.e the input_dir substring is replaced with output_dir.
    all_triplets = []
    all_subdirs.sort()
    for subdir in all_subdirs:
        png_files = glob.glob(os.path.join(subdir, '*.png'))
        jpg_files = glob.glob(os.path.join(subdir, '*.jpg'))
        assert len(png_files) * len(jpg_files) == 0
        file_paths = png_files + jpg_files
        assert len(file_paths) > 0
        file_paths.sort()
        for i in range(len(file_paths) - 2):
            input_a_path = file_paths[i]
            input_b_path = file_paths[i + 1]
            input_c_path = file_paths[i + 2]
            output_b_path = os.path.join(output_dir, os.path.relpath(input_b_path, start=input_dir))
            all_triplets.append((input_a_path, input_c_path, output_b_path))
    return all_triplets


def get_subdirs(input_dir):
    # Grab all the subdirectories. Each directory is assumed to contain a single video sequence.
    print('Globbing...')
    all_png_files = glob.glob(os.path.join(input_dir, '**', '*.png'), recursive=True)
    all_jpg_files = glob.glob(os.path.join(input_dir, '**', '*.jpg'), recursive=True)
    all_file_paths = all_png_files + all_jpg_files
    all_subdirs_set = set()
    for path in all_file_paths:
        subdir = os.path.dirname(path)
        all_subdirs_set.add(subdir)
    all_subdirs = list(all_subdirs_set)
    print('There are %d subdirectories.' % len(all_subdirs))
    return all_subdirs


def add_args(parser):
    parser.add_argument('-d', '--input_directory', type=str,
                        help='Input directory.')
    parser.add_argument('-o', '--output_directory', type=str,
                        help='Output directory.')
    parser.add_argument('-f', '--file_path', type=str,
                        help='Output file that we will write the evaluation paths to.')


def main():
    """
    Writes evaluation paths to an output file. Each line of the file will be a comma-separated tuple, e.g
    'input/path/a.png, input/path/c.png, output/path/b.png'
    """
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    triplet_filepaths = get_triplet_filepaths(args.input_directory, args.output_directory)

    # Save to text file.
    with open(args.file_path, 'w') as f:
        lines = []
        for triplet in triplet_filepaths:
            lines.append('%s, %s, %s\n' % triplet)
        f.writelines(lines)


if __name__ == '__main__':
    main()
