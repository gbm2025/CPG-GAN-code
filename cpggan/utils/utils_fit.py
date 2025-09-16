import os
import torch
import torch.distributed as dist
from tqdm import tqdm

from utils.utils import get_lr, show_result


def fit_one_epoch(G_model_train, D_model_train, G_model, D_model, loss_history, G_optimizer, D_optimizer, BCE_loss,
                  epoch, epoch_step, gen, Epoch, cuda, fp16, scaler, save_period, save_dir, photo_save_step,
                  local_rank=0):
    G_total_loss = 0
    D_total_loss = 0

    if local_rank == 0:
        # print('Start Train')
        # print('\n')
        pbar = tqdm(total=epoch_step, desc=f'Epoch: [{epoch + 1}/{Epoch}]', postfix=dict, mininterval=0.3)
    for iteration, images in enumerate(gen):
        if iteration >= epoch_step:
            break

        batch_size = images.size()[0]
        y_real = torch.ones(batch_size)
        y_fake = torch.zeros(batch_size)
        noise_1 = torch.randn((batch_size, 100))
        noise_2 = torch.randn((batch_size, 100))

        with torch.no_grad():
            if cuda:
                images, y_real, y_fake, noise_1, noise_2 = images.cuda(local_rank), y_real.cuda(
                    local_rank), y_fake.cuda(local_rank), noise_1.cuda(local_rank), noise_2.cuda(local_rank)

        if not fp16:

            # ----------------------------------------------------#
            D_optimizer.zero_grad()

            images = images  #
            y_real = y_real  #

            D_result = D_model_train(images)

            D_result = D_result  #

            D_real_loss = BCE_loss(D_result, y_real)  ##########
            D_real_loss.backward()

            # D_result = D_model_train(images)
            # D_real_loss = BCE_loss(D_result, y_real)  ##########
            # D_real_loss.backward()

            # -----------------------------------

            aaa = y_fake
            ggg = noise_1

            G_result = G_model_train(noise_1)
            G_result = G_result

            D_result = D_model_train(G_result)
            D_fake_loss = BCE_loss(D_result, y_fake)
            D_fake_loss.backward()
            D_optimizer.step()

            # G_result                = G_model_train(noise_1)
            # D_result                = D_model_train(G_result)
            # D_fake_loss             = BCE_loss(D_result, y_fake)
            # D_fake_loss.backward()
            # D_optimizer.step()

            # ----------------------------------------------------#
            # ----------------------------------------------------#
            G_optimizer.zero_grad()
            G_result = G_model_train(noise_2)
            D_result = D_model_train(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real)
            G_train_loss.backward()
            G_optimizer.step()

        else:
            from torch.cuda.amp import autocast

            # ----------------------------------------------------#

            # ----------------------------------------------------#
            with autocast():
                D_optimizer.zero_grad()
                D_result = D_model_train(images)
                D_real_loss = BCE_loss(D_result, y_real)
            # ----------------------#
            # ----------------------#
            scaler.scale(D_real_loss).backward()

            with autocast():
                G_result = G_model_train(noise_1)
                D_result = D_model_train(G_result)
                D_fake_loss = BCE_loss(D_result, y_fake)
            # ----------------------#
            # ----------------------#
            scaler.scale(D_fake_loss).backward()
            scaler.step(D_optimizer)
            scaler.update()

            # ----------------------------------------------------#

            # ----------------------------------------------------#
            with autocast():
                G_optimizer.zero_grad()
                G_result = G_model_train(noise_2)
                D_result = D_model_train(G_result).squeeze()
                G_train_loss = BCE_loss(D_result, y_real)
            # ----------------------#
            # ----------------------#
            scaler.scale(G_train_loss).backward()
            scaler.step(G_optimizer)
            scaler.update()

        G_total_loss += G_train_loss.item()
        D_total_loss += (D_real_loss.item() + D_fake_loss.item()) * 0.5

        if local_rank == 0:
            pbar.set_postfix(**{'G_loss': G_total_loss / (iteration + 1),
                                'D_loss': D_total_loss / (iteration + 1),
                                'lr': get_lr(G_optimizer)})
            pbar.update(1)

            if iteration % photo_save_step == 0:
                show_result(epoch + 1, G_model, cuda, local_rank)

    G_total_loss = G_total_loss / epoch_step
    D_total_loss = D_total_loss / epoch_step

    if local_rank == 0:
        pbar.close()
        print('Epoch: [' + str(epoch + 1) + ' / ' + str(Epoch) + ']')
        print('G_Loss: [%.4f]   ||   D_Loss: [%.4f] ' % (G_total_loss, D_total_loss))
        loss_history.append_loss(epoch + 1, G_total_loss=G_total_loss, D_total_loss=D_total_loss)

        # ----------------------------#
        # ----------------------------#

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(G_model.state_dict(), os.path.join(save_dir, 'G_Epoch_%d.pkl' % (
                    epoch + 1)))
            torch.save(D_model.state_dict(), os.path.join(save_dir, 'D_Epoch_%d.pkl' % (
                    epoch + 1)))

        torch.save(G_model.state_dict(), os.path.join(save_dir, "G_last.pkl"))
        torch.save(D_model.state_dict(), os.path.join(save_dir, "D_last.pkl"))
