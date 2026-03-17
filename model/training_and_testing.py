import torch
import matplotlib.pyplot as plt
import os
import numpt as np

def train(model,
          criterion,
          optimizer,
          epochs,
          train_loader,
          val_loader,
          save_dir,
          **kwargs):
    """
    Training loop
    """
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    start_epoch = kwargs.get('start_epoch', 0)
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        model.train()
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output, __ = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = ((output.argmax(dim=1) == label).float().mean())
            epoch_accuracy += acc/len(train_loader)
            epoch_loss += loss/len(train_loader)

        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        train_loss_history.append(epoch_loss.cpu().detach().numpy())
        train_accuracy_history.append(epoch_accuracy.cpu().detach().numpy())

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy= 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output, __ = model(data)

                val_loss = criterion(val_output,label)
                acc = ((val_output.argmax(dim=1) == label).float().mean())
                epoch_val_accuracy += acc/ len(val_loader)
                epoch_val_loss += val_loss/ len(val_loader)

            print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))
            val_loss_history.append(epoch_val_loss.cpu().detach().numpy())
            val_accuracy_history.append(epoch_val_accuracy.cpu().detach().numpy())

        if (epoch + 1) % 10 == 0 and epoch != 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
            }, os.path.join(save_dir, f'ckpts/model_ckpt_{epoch + 1}_{epoch_val_loss.cpu().detach().numpy():.2f}.tar'))

        # Save updated training curves after each epoch
        plt.figure('Loss')
        plt.plot(train_loss_history, label='train')
        plt.plot(val_loss_history, label='validation')
        plt.xlim([0, epochs])
        plt.legend()
        plt.title('Loss')
        plt.savefig(os.path.join(save_dir, f'loss_curves_from_{start_epoch}.png'))
        plt.close()

        plt.figure('Accuracy')
        plt.plot(train_accuracy_history, label='train')
        plt.plot(val_accuracy_history, label='validation')
        plt.xlim([0, epochs])
        plt.legend()
        plt.title('Accuracy')
        plt.savefig(os.path.join(save_dir, f'accuracy_curves_from_{start_epoch}.png'))
        plt.close()

def test(model, test_loader, classes, save_dir, **kwargs):
    labels = []
    preds = []
    class_probs = []
    test_accuracy = 0
    feat_vecs = []
    test_outputs = []

    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            label = label.to(device)

            test_output, feat_vec = model(data)

            test_pred = (test_output.argmax(dim=1))

            acc = (test_pred == label).float().mean()

            test_accuracy += acc/len(test_loader)

            test_outputs.append(test_output)
            labels.append(label)
            preds.append(test_pred)
            class_probs.append(F.softmax(test_output, dim=1))
            feat_vecs.append(feat_vec)

    test_outputs = torch.cat(test_outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    class_probs = torch.cat(class_probs, dim=0)
    feat_vecs = torch.cat(feat_vecs, dim=0)

    np.savetxt(os.path.join(save_dir, 'feat_vecs.txt'), feat_vecs.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'labels.txt'), labels.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'preds.txt'), preds.cpu().numpy())
    np.savetxt(os.path.join(save_dir, 'test_outputs.txt'), test_outputs.cpu().numpy())
