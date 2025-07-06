"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from __future__ import print_function
import argparse
import random
import os
import numpy as np
import torch

from datetime import datetime
import logging
from Codes import Code
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from qecc_dataset import setup_dataloader, setup_dataloader_list
from train import train, test

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


##################################################################

def setup_model_and_optimizer(args):
    if args.repetitions > 1:
        from Model_T_measurements import ECC_Transformer
    else:
        from Model import ECC_Transformer
    model = ECC_Transformer(args, dropout=0).to(device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    return model, optimizer, scheduler


def training_loop(args, model, optimizer, scheduler, train_dataloader, test_dataloader_list, ps_test):
    best_loss = float('inf')
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc='Training Progress', unit='epoch')
    for epoch in epoch_pbar:
        loss, ber, ler = train(
            model, device, train_dataloader, optimizer,
            epoch, LR=scheduler.get_last_lr()[0], lambda_loss_ber=args.lambda_loss_ber, lambda_loss_ler=args.lambda_loss_ler, lambda_loss_n_pred=args.lambda_loss_n_pred
        )
        scheduler.step()
        torch.save(model, os.path.join(args.path, 'last_model'))
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model'))
            logger.info('Model Saved')
        epoch_pbar.set_postfix({
            'Loss': f'{loss:.3e}',
            'BER': f'{ber:.3e}',
            'LER': f'{ler:.3e}',
            'Best_Loss': f'{best_loss:.3e}'
        })
        if epoch % 60 == 0 or epoch in [1, args.epochs]:
            test(model, device, test_dataloader_list, ps_test)
    epoch_pbar.close()



def main(args):
    """Main training and evaluation loop."""
    args.code.logic_matrix = args.code.logic_matrix.to(device)
    args.code.pc_matrix = args.code.pc_matrix.to(device)

    assert args.repetitions > 0, "Repetitions must be positive."

    logger.info(f'PC matrix shape {args.code.pc_matrix.shape}')

    ps_test = np.linspace(0.001, 0.01, 15)
    if args.noise_type == 'depolarization' or args.repetitions > 1:
        ps_test = np.linspace(0.001, 0.01, 1)
    ps_train = ps_test

    model, optimizer, scheduler = setup_model_and_optimizer(args)
    logger.info(model)
    logger.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')

    train_dataloader = setup_dataloader(args.code, args.noise_type, args.repetitions, seed=args.seed, batch_size=args.batch_size, workers=args.workers, ps=ps_train)
    test_dataloader_list = setup_dataloader_list(args.code, args.noise_type, args.repetitions, seed=args.seed, batch_size=args.test_batch_size, workers=args.workers, ps=ps_test)
    training_loop(args, model, optimizer, scheduler, train_dataloader, test_dataloader_list, ps_test)
 
    # Final evaluation
    print("FINAL EVALUATION")
    print("=" * 50)
    model = torch.load(os.path.join(args.path, 'best_model')).to(device)
    logger.info('Best model loaded')
    test(model, device, test_dataloader_list, ps_test)


##################################################################################################################
##################################################################################################################
##################################################################################################################

def setup_logging(model_dir):
    """Set up logging to file and stdout."""
    handlers = [logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='PyTorch DQEC')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpus', type=str, default='0', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='honeycomb', choices=['toric', 'honeycomb'])
    parser.add_argument('--code_L', type=int, default=3, help='Lattice length')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of faulty repetitions. <=1 is equivalent to none.')
    parser.add_argument('--noise_type', type=str, default='independent', choices=['independent', 'depolarization', 'circuit'],
                        help='Noise model')

    # model args
    parser.add_argument('--N_dec', type=int, default=6, help='Number of QECCT self-attention modules')
    parser.add_argument('--d_model', type=int, default=128, help='QECCT dimension')
    parser.add_argument('--h', type=int, default=16, help='Number of heads')

    # qecc args
    parser.add_argument('--lambda_loss_ber', type=float, default=0.5, help='BER loss regularization')
    parser.add_argument('--lambda_loss_ler', type=float, default=1., help='LER loss regularization')
    parser.add_argument('--lambda_loss_n_pred', type=float, default=0.5, help='g noise prediction regularization')

    # ablation args
    parser.add_argument('--no_g', type=int, default=0)
    parser.add_argument('--no_mask', type=int, default=0)

    return parser.parse_args()


def main_entry():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################
    if args.no_g > 0:
        args.lambda_loss_n_pred = 0.
    ###
    args.code = Code(args.code_type, args.code_L, args.noise_type)

    model_dir = os.path.join('Final_Results_QECCT', args.code_type,
                             'Code_L_' + str(args.code_L),
                             f'noise_model_{args.noise_type}',
                             f'repetition_{args.repetitions}',
                             datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    os.makedirs(model_dir, exist_ok=True)

    setup_logging(model_dir)
    logger.info(f"Path to model/logs: {model_dir}")
    logger.info(args)

    args.path = model_dir

    main(args)


if __name__ == '__main__':
    main_entry()