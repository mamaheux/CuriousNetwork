import os
import imageio

for root, subdirs, files in os.walk('../../database/mp4'):

    print(root)

    file_index = 0
    for filename in files:
        if not filename.lower().endswith('mp4'):
            continue

        input_file_path = os.path.join(root, filename)
        output_dir_path = root.replace('mp4', 'jpg')
        os.makedirs(output_dir_path, exist_ok=True)

        print('\t', filename)

        frame_index = 0
        with imageio.get_reader(input_file_path, 'ffmpeg') as video_reader:
            for image in video_reader:
                imageio.imwrite(os.path.join(output_dir_path, str(file_index) + '-' + str(frame_index) + '.jpg'), image)
                frame_index += 1

        file_index += 1

