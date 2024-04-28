from datasets import TrackNetDataset
import torch
from tracknet import BallTrackNet
import os
from general import train, validate

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    
    NUM_EPOCHS = 150
    BATCH_SIZE = 4

    train_dataset = TrackNetDataset(mode='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    test_dataset = TrackNetDataset(mode='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model = BallTrackNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    exps_path = 'exps'
    plots_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    log_writer = SummaryWriter('exps/plots')
    model_last_path = os.path.join(exps_path, 'model_last.pth')
    model_best_path = os.path.join(exps_path, 'model_best.pth')

    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
    val_best_metric = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, device, epoch)
        print('train_loss = {}'.format(train_loss))
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)
        log_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch > 0) and (epoch % 20 == 0):
            val_loss, precision, recall, f1 = validate(model, test_loader, device, epoch)
            print('val_loss = {}'.format(val_loss))
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/recall', recall, epoch)
            log_writer.add_scalar('Val/f1', f1, epoch)
            if f1 > val_best_metric:
                val_best_metric = f1
                torch.save(model.state_dict(), model_best_path)
            torch.save(model.state_dict(), model_last_path)