
from __future__ import absolute_import, division, print_function

import os
import logging
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.optim import Adam
import matplotlib

from util import *
from graph_model import EHROntologyModel
from data_loader import load_dataset

matplotlib.use("Agg")


def test(args, model, test_dataloader, cohorts, random_state=0, visualize=False, train_dataloader=None, train_cohorts=None, k=1, device='cpu'):

    def get_embs(model, dataloader, device):

        all_embs = []
        iterator = tqdm(dataloader)
        for batch in iterator:
            with torch.no_grad():
                model.eval()
                outputs = model(left_x=batch.x.to(device),
                                left_graph_index=batch.edge_index.long().to(device),
                                left_x_batch=batch.batch.to(device),
                                left_edge_type=batch.edge_type.to(device))
                all_embs.append(outputs.detach().cpu().numpy())

        embs = np.concatenate(all_embs)
        return embs
    # Perform clustering test.

    def cluster_test():

        embs = get_embs(model, test_dataloader, device)
        me_purity, me_nmi, me_ri = cal_cluster_metric(
            embs, cohorts, random_state)
        if visualize:
            plot_embedding(embs, cohorts, args.use_conv, args.output_dir)

        return {'purity': me_purity, 'nmi': me_nmi, 'ri': me_ri}

    # Perform disease prediction test.
    def knn_test():
        """
        Perform k-NN test.

        Args:
            model (torch.nn.Module): Model for generating embeddings.
            test_dataloader (DataLoader): DataLoader for the test dataset.
            cohorts: Cohorts.
            train_dataloader (DataLoader): DataLoader for the training dataset.
            train_cohorts: Cohorts for training data.
            device: Device for computation.

        Returns:
            dict: Dictionary containing k-NN results.

        """

        embs = get_embs(model, test_dataloader, device)
        train_embs = get_embs(model, train_dataloader, device)

        '''
        res = {}
        for k in [1, 3, 5, 7, 9]:
            res[k] = get_top_k_results(
                embs, train_embs, cohorts, train_cohorts, k=k)
        '''

        res = get_top_k_results(
            embs, train_embs, cohorts, train_cohorts, k=k)
        return res

    if args.task == 'cluster':
        res = cluster_test()

    elif args.task == 'knn':
        res = knn_test()
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', default=False,
                        action='store_true', help='whether to train the model.')

    parser.add_argument('--do_test', default=False, action='store_true',
                        help='whether to test the performance on test dataset.')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='the initial learning rate for Adam.')

    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2,
                        help='the droput rate for self attention transformer.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for train dataset')
    parser.add_argument('--gcn_conv_nums', type=int, default=2,
                        help='the number of gcn convoluation  layer for our model')
    parser.add_argument('--epoch', type=int, default=10,
                        help='the number of training epoch')
    parser.add_argument('--emb_size', type=int, default=100,
                        help='the vocab embedding size')
    parser.add_argument('--hidden_size', type=int, default=200,
                        help='the ontology embedding hidden size')
    parser.add_argument('--graph_heads', type=int,
                        default=1, help='the graph heads')

    parser.add_argument('--out_channels', type=int, default=300,
                        help='the output channels for graph model')
    parser.add_argument('--seed', type=int, default=1002,
                        help='random seed for initialization')
    parser.add_argument('--pair_neurons', type=int, default=10,
                        help='the neurons for global embedding')
    parser.add_argument('--use_conv', type=str, default='rgcn',
                        help='the convolution method : gcn, gin, gat, gated_conv')
    parser.add_argument('--train_data', type=str, required=False,
                        help='the processed data  path for train_data.pt ')
    parser.add_argument('--test_cluster_data', type=str, required=False,
                        help='the processed data  path for test_data.pt')
    parser.add_argument('--test_knn_data', type=str, required=False,
                        help='the processed data  path for test_wout_diag.pt')
    parser.add_argument('--valid_knn_data', type=str, required=False,
                        help='the processed data  path for valid_wout_diag.pt')
    parser.add_argument('--valid_cluster_data', type=str, required=False,
                        help='the processed data  path for valid_cluster.pt')

    parser.add_argument('--train_input_data', type=str, required=False,
                        help='the processed data  path for train dataset embedding ')

    parser.add_argument('--dataset', type=str,
                        required=True, help='mimic3 or mimic4')
    parser.add_argument('--task', type=str, required=True,
                        help='cluster or knn')
    parser.add_argument('--resume_path', type=str, required=False,
                        help='the path for resume model')
    parser.add_argument("--output_dir",
                        default='saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('loading dataset')

    # load dataset
    vocab_emb, tokenizer, train_dataloader, test_dataloader, valid_cluster_dataloader, test_knn_dataloader, valid_knn_dataloader, \
        test_cohorts, valid_cohorts, train_input_dataloader, train_cohorts = load_dataset(args, args.train_data,
                                                                                          args.test_cluster_data, args.valid_cluster_data, args.test_knn_data,
                                                                                          args.valid_knn_data, args.train_input_data, args.batch_size)

    diag_voc, proce_voc, pres_voc = tokenizer.diag_voc, tokenizer.proce_voc, tokenizer.atc_voc

    model = EHROntologyModel(args, vocab_emb, diag_voc,
                             proce_voc, pres_voc, device).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # Create directory for output
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(args, os.path.join(
        args.output_dir, 'config_'+args.task+'.txt'))
    model_to_save = model.module if hasattr(model, 'module') else model

    if args.task == 'cluster':
        model_name = 'pytorch_cluster.bin'
    elif args.task == 'knn':
        model_name = 'pytorch_prediction.bin'
    else:
        raise ValueError(
            "Invalid task argument. Supported values are 'cluster' and 'knn'.")

    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

    log_dir = os.path.join(args.output_dir, 'logs', args.task)
    os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(
        log_dir, get_time_string()+'.txt'), encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if args.do_test:
        model.load_state_dict(torch.load(args.resume_path))
        model.eval()
        res = test(args, model, test_dataloader if args.task == 'cluster' else test_knn_dataloader,
                   test_cohorts, visualize=True, train_dataloader=train_input_dataloader if args.task == 'knn' else None,
                   train_cohorts=train_cohorts if args.task == 'knn' else None, device=device)

        for key, value in res.items():
            print(f'{key}: {value}')
        return

    if args.do_train:
        global_step = 0

        best_metric = 0
        best_epoch = 0

        # Define the metric to be used based on the task
        if args.task == 'cluster':
            metric_name = 'purity'
        elif args.task == 'knn':
            metric_name = 'accu'

        for epoch in trange(int(args.epoch), desc='training epoch'):

            logger.info(
                f'**************************** {epoch} ****************************')

            tr_loss, nb_tr_steps = 0, 0
            epoch_iterator = tqdm(
                train_dataloader, leave=False, desc='Training')

            all_p_labels = [] # List to store all predicted labels 
            all_p_preds = []# List to store all predicted probabilities
            model.train()

            for batch in epoch_iterator:
                loss, p_output = model(left_x=batch.x_left.to(device),
                                       left_graph_index=batch.edge_index_left.long().to(device),
                                       left_x_batch=batch.x_left_batch.to(
                                           device),
                                       right_x=batch.x_right.to(device),
                                       right_graph_index=batch.edge_index_right.long().to(device),
                                       right_x_batch=batch.x_right_batch.to(
                                           device),
                                       label=batch.y.long().to(device),
                                       left_edge_type=batch.left_edge_type.to(
                                           device),
                                       right_edge_type=batch.right_edge_type.to(
                                           device),
                                       )

                predict_score = p_output.detach().cpu().numpy()
                p_labels = np.argmax(predict_score, axis=1)
                loss.backward()
                all_p_preds.append(p_labels)
                all_p_labels.append(batch.y.long().detach().cpu().numpy())

                tr_loss += loss.item()
                nb_tr_steps += 1

                epoch_iterator.set_postfix(loss='%.4f' % (tr_loss/nb_tr_steps))

                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            logger.info(
                '**************************** Running eval ****************************')
            logger.info("num of examples = %d", len(
                valid_cluster_dataloader)*args.batch_size)
            model.eval()
            res = test(args, model, valid_cluster_dataloader if args.task == 'cluster' else valid_knn_dataloader,
                       valid_cohorts, train_dataloader=train_input_dataloader if args.task == 'knn' else None,
                       train_cohorts=train_cohorts if args.task == 'knn' else None, device=device)

            # Update best metric and epoch if the current metric is better
            if res[metric_name] > best_metric:
                best_metric = res[metric_name]
                best_epoch = epoch
                torch.save(model_to_save.state_dict(),
                           os.path.join(args.output_dir, model_name))

            log_result(logger, best_epoch, res)


if __name__ == '__main__':
    main()
