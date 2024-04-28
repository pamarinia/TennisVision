from itertools import groupby
import cv2
import numpy as np
import torch

from general import postprocess
from scipy.spatial import distance

from tracknet import BallTrackNet

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def infer_model(frames, model):
    ball_track = [(None, None)]*2
    dists = [-1]*2
    for num in range(2, len(frames)):
        img_1 = cv2.resize(frames[num-2], (640, 360))
        img_2 = cv2.resize(frames[num-1], (640, 360))
        img_3 = cv2.resize(frames[num], (640, 360))
        imgs = np.concatenate((img_3, img_2, img_1), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        imgs = np.expand_dims(imgs, axis=0)
        input = torch.from_numpy(imgs).float().to(device)

        out = model(input)
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)

        ball_track.append((x_pred, y_pred))
        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)
    return ball_track, dists


def remove_outliers(ball_track, dists, max_dist=100):
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track

# def split_track(ball_track, max_gap=10, max_dist_gap=80, min_track=5):
#     # 0 - ball detected, 1 - ball not detected
#     list_det = [0 if x[0] else 1 for x in ball_track]
#     # Split the track into groups of (k=0 or 1, len(subgroup g))
#     groups = [(k, sum(1 for i in g)) for k, g in groupby(list_det)]
    
#     cursor = 0
#     min_value = 0
#     result = []
#     for i, (k, l) in enumerate(groups):
#         # If the group is a ball not detected group
#         if (k == 1) and (i > 0) and (i < len(groups) - 1):
#             # Calculate the distance between the previous detected ball and the next detected ball
#             dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor + l])
#             # If there is too much non detected ball in a row or if dist is greater than the max_dist_gap/l
#             if (l >= max_gap) | (dist/l > max_dist_gap):
#                 # If the gap between the detected balls is greater than the min_track
#                 if (cursor - min_value) > min_track:
#                     result.append([min_value, cursor])
#                     min_value = cursor + l - 1 
#         cursor += l
#     if len(list_det) - min_value > min_track:
#         result.append([min_value, len(list_det)])
#     return result


def interpolation(coords):

    def nan_helper(y):
        # Helper to handle indices and logical indices of NaNs.
        return np.isnan(y), lambda z: z.nonzero()[0]
    
    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    # Interpolate the missing values
    nans, xx = nan_helper(x)
    x[nans] = np.interp(xx(nans), xx(~nans), x[~nans])
    x = np.rint(x)
    nans, yy = nan_helper(y)
    y[nans] = np.interp(yy(nans), yy(~nans), y[~nans])
    y = np.rint(y)

    return list(zip(x, y))



def write_track(frames, ball_track, path_output_video, fps, trace=7):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for num in range(len(frames)):
        frame = frames[num]
        for i in range(trace):
            if (num-i > 0):
                if ball_track[num-i][0]:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    frame = cv2.circle(frame, (x, y), radius=0, color=(0, 255, 0), thickness=10-i)
                else:
                    break
        out.write(frame)
    out.release()


if __name__ == '__main__':

    model = BallTrackNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.load_state_dict(torch.load('exps/model_best.pth'))
    model = model.to(device)
    model.eval()

    frames, fps = read_video('input/Med_Djo_cut.mp4')
    ball_track, dists = infer_model(frames, model)
    #print(ball_track)

    ball_track = remove_outliers(ball_track, dists)
    

    #subtracks = split_track(ball_track)
    # for subtrack in subtracks:
    #     ball_subtrack = ball_track[subtrack[0]:subtrack[1]]
    #     ball_subtrack = interpolation(ball_subtrack)
    #     ball_track[subtrack[0]:subtrack[1]] = ball_subtrack
    
    ball_track = interpolation(ball_track)
    print(ball_track)
    write_track(frames, ball_track, 'outputs/Med_Djo_cut_tracked.avi', fps)




