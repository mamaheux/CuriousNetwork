import os
import imageio

for root, subdirs, files in os.walk('../../database/mp4'):
    i = 0

    print(root)

    for filename in files:
        if not filename.lower().endswith('mp4'):
            continue

        input_file_path = os.path.join(root, filename)
        output_dir_path = root.replace('mp4', 'jpg')
        os.makedirs(output_dir_path, exist_ok=True)

        print('\t', filename)

        with imageio.get_reader(input_file_path, 'ffmpeg') as video_reader:
            for image in video_reader:
                imageio.imwrite(os.path.join(output_dir_path, str(i) + '.jpg'), image)
                i += 1
