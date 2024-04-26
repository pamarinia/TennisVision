import numpy as np
import pandas as pd
import os
import cv2


def gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = 255 * np.exp(-(x**2 + y**2)/float(2*variance))
    return g.astype(int)

def create_ground_truth_images(input_path, output_path, size, variance, height, width):
    gaussian_kernel_array = gaussian_kernel(size, variance)
    for game_id in range(1, 11):
        game = 'game{}'.format(game_id)
        clips = os.listdir(os.path.join(input_path, game))
        for clip in clips:
            print('game: {}, clip: {}'.format(game, clip))

            # Create the directory structure for game
            path_out_game = os.path.join(output_path, game)
            if not os.path.exists(path_out_game):
                os.makedirs(path_out_game)

            # Create the directory structure for clip
            path_out_clip = os.path.join(path_out_game, clip)
            if not os.path.exists(path_out_clip):
                os.makedirs(path_out_clip)

            # Load the labels
            path_labels = os.path.join(input_path, game, clip, 'Label.csv')
            labels = pd.read_csv(path_labels)
            for idx in range(labels.shape[0]):
                # Extract the label information
                file_name, visibility, x, y, status = labels.iloc[idx]
                # Create the heatmap
                heatmap = np.zeros((height, width), dtype=np.uint8)

                if visibility != 0:
                    x = int(x)
                    y = int(y)
                    for i in range(-size, size+1):
                        for j in range(-size, size+1):
                            if x + i >= 0 and x + i < width and y + j >= 0 and y + j < height:
                                heatmap[y + j, x + i] = gaussian_kernel_array[j + size, i + size]

                cv2.imwrite(os.path.join(path_out_clip, file_name), heatmap)

def create_ground_truth_labels(input_path, output_path, train_rate=0.7):
    df = pd.DataFrame()
    for game_id in range(1, 11):
        game = 'game{}'.format(game_id)
        clips = os.listdir(os.path.join(input_path, game))
        for clip in clips:
            # Load the labels
            path_labels = os.path.join(input_path, game, clip, 'Label.csv')
            labels = pd.read_csv(path_labels)
            labels['img_path_1'] = 'images/' + game +'/' + clip + '/' + labels['file name']
            labels['ground_truth_path'] = 'ground_truth/' + game +'/' + clip + '/' + labels['file name']
            
            # We startat the third image to have 3 consecutive images
            labels_target = labels[2:].copy()
            labels_target.loc[:, 'img_path_2'] = list(labels['img_path_1'][1:-1])
            labels_target.loc[:, 'img_path_3'] = list(labels['img_path_1'][:-2])
            df = pd.concat([df, labels_target])
    
    df = df.reset_index(drop=True)
    df = df[['img_path_1', 'img_path_2', 'img_path_3', 'ground_truth_path', 'x-coordinate', 'y-coordinate', 'visibility', 'status']]
    
    # We shuffle the data
    df = df.sample(frac=1)
    num_train = int(train_rate * df.shape[0])
    df_train = df[:num_train]
    df_test = df[num_train:]
    df_train.to_csv(os.path.join(output_path, 'labels_train.csv'), index=False)
    df_test.to_csv(os.path.join(output_path, 'labels_test.csv'), index=False)
