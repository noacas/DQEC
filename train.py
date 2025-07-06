import time
import torch
from tqdm import tqdm
import logging
import numpy as np

from Codes import sign_to_bin, bin_to_sign, BER, FER

logger = logging.getLogger(__name__)


def diff_GF2_mul(H, x):
    H_bin = sign_to_bin(H) if -1 in H else H
    x_bin = x

    tmp = bin_to_sign(H_bin.unsqueeze(0) * x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp, 1)
    tmp = sign_to_bin(tmp)
    return tmp


def logical_flipped(L, x):
    return torch.matmul(x.float(), L.float()) % 2


def train(model, device, train_loader, optimizer,
          epoch, LR, lambda_loss_ber,
          lambda_loss_n_pred, lambda_loss_ler):
    model.train()
    cum_loss = cum_ber = cum_ler = cum_samples = 0
    cum_loss1 = cum_loss2 = cum_loss3 = 0
    t = time.time()
    bin_fun = torch.sigmoid

    # Create progress bar for training batches
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=True)

    for batch_idx, (x, z, y, magnitude, syndrome) in enumerate(pbar):
        syndrome = syndrome.to(device)
        z_pred = model(magnitude.to(device), syndrome)
        loss1, loss2 = model.module.loss(-z_pred, z.to(device))
        loss3 = torch.nn.functional.binary_cross_entropy_with_logits(
            (diff_GF2_mul(train_loader.dataset.logic_matrix, bin_fun(-z_pred))),
            logical_flipped(train_loader.dataset.logic_matrix, z.to(device)))
        loss = lambda_loss_ber * loss1 + lambda_loss_n_pred * loss2 + lambda_loss_ler * loss3
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        z_pred = sign_to_bin(torch.sign(-z_pred))
        ber = BER(z_pred, z.to(device))
        ler = FER(logical_flipped(train_loader.dataset.logic_matrix, z_pred),
                  logical_flipped(train_loader.dataset.logic_matrix, z.to(device)))

        cum_loss += loss.item() * z.shape[0]
        #
        cum_loss1 += loss1.item() * z.shape[0]
        cum_loss2 += loss2.item() * z.shape[0]
        cum_loss3 += loss3.item() * z.shape[0]
        #
        cum_ber += ber * z.shape[0]
        cum_ler += ler * z.shape[0]
        cum_samples += z.shape[0]

        # Update progress bar with current metrics
        current_loss = cum_loss / cum_samples
        current_ber = cum_ber / cum_samples
        current_ler = cum_ler / cum_samples

        pbar.set_postfix({
            'Loss': f'{current_loss:.3e}',
            'BER': f'{current_ber:.3e}',
            'LER': f'{current_ler:.3e}',
            'LR': f'{LR:.2e}'
        })

        #
        if (batch_idx + 1) % (len(train_loader) // 2) == 0 or batch_idx == len(train_loader) - 1:
            logger.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.5e} BER={cum_ber / cum_samples:.3e} LER={cum_ler / cum_samples:.3e}')
            logger.info(
                f'***Loss={cum_loss / cum_samples:.5e} Loss LER={cum_loss3 / cum_samples:.5e} Loss BER={cum_loss1 / cum_samples:.5e} Loss noise pred={cum_loss2 / cum_samples:.5e}')

    pbar.close()
    logger.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_ler / cum_samples


def test(model, device, test_loader_list, ps_range_test, cum_count_lim=100000):
    model.eval()
    test_loss_ber_list, test_loss_ler_list, cum_samples_all = [], [], []
    t = time.time()

    with torch.no_grad():
        # Create progress bar for test noise levels
        test_pbar = tqdm(enumerate(test_loader_list), total=len(test_loader_list),
                         desc='Testing', leave=True)

        for ii, test_loader in test_pbar:
            test_ber = test_ler = cum_count = 0.

            # Create inner progress bar for samples at current noise level
            sample_pbar = tqdm(total=cum_count_lim, desc=f'p={ps_range_test[ii]:.3e}',
                               leave=False, unit='samples')

            while True:
                (x, z, y, magnitude, syndrome) = next(iter(test_loader))
                z_pred = model(magnitude.to(device), syndrome.to(device))
                _ = model.module.loss(-z_pred, z.to(device))
                z_pred = sign_to_bin(torch.sign(-z_pred))

                test_ber += BER(z_pred, z.to(device)) * z.shape[0]
                test_ler += FER(logical_flipped(test_loader.dataset.logic_matrix, z_pred),
                                logical_flipped(test_loader.dataset.logic_matrix, z.to(device))) * z.shape[0]
                cum_count += z.shape[0]

                # Update sample progress bar
                sample_pbar.update(z.shape[0])
                current_ber = test_ber / cum_count
                current_ler = test_ler / cum_count
                sample_pbar.set_postfix({
                    'BER': f'{current_ber:.3e}',
                    'LER': f'{current_ler:.3e}'
                })

                if cum_count > cum_count_lim:
                    break

            sample_pbar.close()

            cum_samples_all.append(cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_ler_list.append(test_ler / cum_count)

            # Update main test progress bar
            test_pbar.set_postfix({
                'Current_BER': f'{test_loss_ber_list[-1]:.3e}',
                'Current_LER': f'{test_loss_ler_list[-1]:.3e}'
            })

            print(f'Test p={ps_range_test[ii]:.3e}, BER={test_loss_ber_list[-1]:.3e}, LER={test_loss_ler_list[-1]:.3e}')

        test_pbar.close()

        ###
        logger.info('Test LER  ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ler_list, ps_range_test))]))
        logger.info('Test BER  ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, ps_range_test))]))
        logger.info(f'Mean LER = {np.mean(test_loss_ler_list):.3e}, Mean BER = {np.mean(test_loss_ber_list):.3e}')
    logger.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_ber_list, test_loss_ler_list