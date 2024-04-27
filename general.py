import torch.nn as nn
import torch
import numpy as np
import cv2
from scipy.spatial import distance

def train(model, train_loader, optimizer, device, epoch, max_iters=200):
    losses = []
    criterion = nn.CrossEntropyLoss()
    for iter_id, batch in enumerate(train_loader):
        
        model.train()
        out = model(batch[0].float().to(device))
        ground_truth = torch.tensor(batch[1], dtype=torch.long, device=device)
        loss = criterion(out, ground_truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('train | epoch = {}, iter = [{}/{}], loss = {}'.format(epoch, iter_id, max_iters, loss.item()))
        losses.append(loss.item())

        if iter_id >= max_iters:
            break

    return np.mean(losses)

def validate(model, test_loader, device, epoch, min_dist=5):
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for iter_id, batch in enumerate(test_loader):
        with torch.no_grad():
            out = model(batch[0].float().to(device))
            ground_truth = torch.tensor(batch[1], dtype=torch.long, device=device)
            loss = criterion(out, ground_truth)
            losses.append(loss.item())

            output = out.argmax(dim=1).detach().cpu().numpy()
            for i in range(len(output)):
                x_pred, y_pred = postprocess(output[i])
                x_gt = batch[2][i]
                y_gt = batch[3][i]
                visibility = batch[4][i]
                if x_pred:
                    if visibility != 0:
                        dist = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dist < min_dist:
                            tp[visibility] +=1
                        else:
                            fp[visibility] +=1
                    else:
                        fp[visibility] +=1
                if not x_pred:
                    if visibility != 0:
                        fn[visibility] +=1
                    else:
                        tn[visibility] +=1
            print('val | epoch = {}, iter = [{}|{}], loss = {}, tp = {}, tn = {}, fp = {}, fn = {} '.format(epoch,
                                                                                                            iter_id,
                                                                                                            len(test_loader),
                                                                                                            round(np.mean(losses), 6),
                                                                                                            sum(tp),
                                                                                                            sum(tn),
                                                                                                            sum(fp),
                                                                                                            sum(fn)))
    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    vc1 = tp[1] + fp[1] + tn[1] + fn[1]
    vc2 = tp[2] + fp[2] + tn[2] + fn[2]
    vc3 = tp[3] + fp[3] + tn[3] + fn[3]
    recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    print('precision = {}'.format(precision))
    print('recall = {}'.format(recall))
    print('f1 = {}'.format(f1))

    return np.mean(losses), precision, recall, f1
    

def postprocess(feature_map, scale=2):
    feature_map *= 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)

    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0]*scale
            y = circles[0][0][1]*scale
    
    return x, y