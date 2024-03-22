import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import os
from util import read_pkl
from build_tree import Voc


class EHRTokenizer(object):
    def __init__(self, vocab_file):
        self.vocab = Voc()  # fused all codes such as diag_codes, proce_codes and atc_codes
        self.diag_voc, self.proce_voc, self.atc_voc = self.add_vocab(
            vocab_file)  # get for build ontology EHR Model

    def add_vocab(self, vocab_file):
        voc1, voc2, voc3 = Voc(), Voc(), Voc()
        all_codes_dic = pickle.load(open(vocab_file, 'rb'))
        diag_codes, proce_codes, atc_codes = ['d_'+d for d in all_codes_dic['diag_codes']], [
            'p_'+p for p in all_codes_dic['proce_codes']], ['a_'+a for a in all_codes_dic['atc_codes']]

        voc1.add_sentence(diag_codes)
        self.vocab.add_sentence(diag_codes)

        voc2.add_sentence(proce_codes)
        self.vocab.add_sentence(proce_codes)

        voc3.add_sentence(atc_codes)
        self.vocab.add_sentence(atc_codes)

        return voc1, voc2, voc3

    # for each single graph transform the code to index
    def build_single_graph(self, diag_codes, proce_codes, atc_codes):
        single_voc = Voc()

        single_voc.add_sentence(['d_'+d for d in diag_codes])
        single_voc.add_sentence(['p_'+p for p in proce_codes])
        single_voc.add_sentence(['a_'+a for a in atc_codes])

        sorted_idx = sorted(single_voc.idx2word.keys())
        sorted_codes = []
        for idx in sorted_idx:
            code = single_voc.idx2word[idx]
            sorted_codes.append(code)

        return single_voc, self.convert_codes_to_ids(sorted_codes, '')

    # construct the graph with ontology code index
    def build_onto_single_graph(self, diag_codes, proce_codes, atc_codes):
        single_voc = Voc()
        single_voc.add_sentence(['d_'+d for d in diag_codes])
        single_voc.add_sentence(['p_'+p for p in proce_codes])
        single_voc.add_sentence(['a_'+a for a in atc_codes])

        code_ids = []

        for code in diag_codes:
            code_ids.append(self.vocab.word2idx['d_'+code])

        for code in proce_codes:
            code_ids.append(self.vocab.word2idx['p_'+code])

        for code in atc_codes:
            if code != 'nan':
                code_ids.append(self.vocab.word2idx['a_'+code])

        return single_voc, code_ids

    def convert_codes_to_ids(self, codes, c_type):
        ids = []
        for code in codes:
            ids.append(self.vocab.word2idx[c_type+code])
        return ids


class UndirectPatientOntoGraphEx(object):
    def __init__(self, diag_codes, proce_codes, atc_codes, rel_infos, tokenizer, disease_prediction=False):
        self.diag_codes = diag_codes
        self.proce_codes = proce_codes
        self.atc_codes = atc_codes
        self.tokenizer = tokenizer

        self.diag_proce_rel, self.diag_atc_rel, self.proce_atc_rel = rel_infos

        self.x, self.edge_index, self.edge_type = \
            self.build_patient_graph(
                self.tokenizer, diag_codes, proce_codes, atc_codes, disease_prediction=disease_prediction)

    # construct patient graph onto_code_ids for ontology index mapping
    def build_patient_graph(self, tokenizer: EHRTokenizer, diag_codes, proce_codes, atc_codes, disease_prediction=False):

        # disease prediciton task only have two type of nodes, proce and atc nodes
        if disease_prediction:
            diag_codes = []

        single_voc, onto_code_ids = tokenizer.build_onto_single_graph(
            diag_codes, proce_codes, atc_codes)

        edge_idx = []

        '''
        for heterogeneous graph neural network model: RGCN, RGAT...
        0,1,2,3,4,5,6,7,8 represent the edge type for diag-diag, proce-proce, atc-atc, 
        diag-proce, proce-diag, diag-atc, atc-diag, proce-atc, atc-proce
        '''
        edge_type = []
        edge_len = len(edge_idx)
        code_set = set()

        # construct the edge for diagnosis and procedure
        all_diag_proce_pairs = [(d, p)
                                for d in diag_codes for p in proce_codes]
        # if there has the relations, construct the edge in this graph
        valid_diag_proce_pairs = [edge_idx.extend([(single_voc.word2idx['d_'+d_p[0]], single_voc.word2idx['p_'+d_p[1]]),
                                                   (single_voc.word2idx['p_'+d_p[1]], single_voc.word2idx['d_'+d_p[0]])])
                                  for d_p in all_diag_proce_pairs if d_p[0]+'-'+d_p[1] in self.diag_proce_rel]
        # update edge type for each edge
        edge_type.extend([3, 4]*int((len(edge_idx)-edge_len)/2))

        for d_p in all_diag_proce_pairs:
            if d_p[0]+'-'+d_p[1] in self.diag_proce_rel:
                if d_p[0] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['d_'+d_p[0]], single_voc.word2idx['d_'+d_p[0]])])
                    code_set.add(d_p[0])
                    edge_type.append(0)

                if d_p[1] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['p_'+d_p[1]], single_voc.word2idx['p_'+d_p[1]])])
                    code_set.add(d_p[1])
                    edge_type.append(1)
        edge_len = len(edge_idx)

        # construct the edge for diagnosis and atc
        all_diag_atc_pairs = [(d, a) for d in diag_codes for a in atc_codes]

        valid_diag_atc_pairs = [edge_idx.extend([(single_voc.word2idx['d_'+d_a[0]], single_voc.word2idx['a_'+d_a[1]]),
                                                 (single_voc.word2idx['a_'+d_a[1]], single_voc.word2idx['d_'+d_a[0]])])
                                for d_a in all_diag_atc_pairs if d_a[0]+'-'+d_a[1] in self.diag_atc_rel]

        edge_type.extend([5, 6]*int((len(edge_idx)-edge_len)/2))

        for d_a in all_diag_atc_pairs:
            if d_a[0]+'-'+d_a[1] in self.diag_atc_rel:
                if d_a[0] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['d_'+d_a[0]], single_voc.word2idx['d_'+d_a[0]])])
                    code_set.add(d_a[0])
                    edge_type.append(0)
                if d_a[1] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['a_'+d_a[1]], single_voc.word2idx['a_'+d_a[1]])])
                    code_set.add(d_a[1])
                    edge_type.append(2)
        edge_len = len(edge_idx)

        # construct the edge for procedure and atc
        all_proce_atc_pairs = [(p, a) for p in proce_codes for a in atc_codes]
        valid_proce_atc_pairs = [edge_idx.extend([(single_voc.word2idx['p_'+p_a[0]], single_voc.word2idx['a_'+p_a[1]]), (single_voc.word2idx['a_'+p_a[1]],
                                                 single_voc.word2idx['p_'+p_a[0]])]) for p_a in all_proce_atc_pairs if p_a[0]+'-'+p_a[1] in self.proce_atc_rel]

        edge_type.extend([7, 8]*int((len(edge_idx)-edge_len)/2))

        for p_a in all_proce_atc_pairs:
            if p_a[0]+'-'+p_a[1] in self.proce_atc_rel:
                if p_a[0] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['p_'+p_a[0]], single_voc.word2idx['p_'+p_a[0]])])
                    code_set.add(p_a[0])
                    edge_type.append(1)
                if p_a[1] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['a_'+p_a[1]], single_voc.word2idx['a_'+p_a[1]])])
                    code_set.add(p_a[1])
                    edge_type.append(2)

        row = list(map(lambda x: x[0], edge_idx))
        col = list(map(lambda x: x[1], edge_idx))

        assert len(row) == len(edge_type)

        return onto_code_ids, [row, col], edge_type

class EHRPairData(Data):
    """
    The data for pairs of EHR graphs
    """

    def __init__(self, x_left, edge_index_left, x_right, edge_index_right, y, left_edge_type,
                 right_edge_type):
        """
        Args:
        x_left (Tensor): Nodes in the left EHR graph.
        x_right (Tensor): Nodes in the right EHR graph.
        edge_index_left (LongTensor): Edge indices of the left EHR graph.
        edge_index_right (LongTensor): Edge indices of the right EHR graph.
        left_edge_type (Tensor): Edge type for edges in the left EHR graph.
        right_edge_type (Tensor): Edge type for edges in the right EHR graph.
        y (Tensor): Label for the EHR pair. 1 for similar, 0 for dissimilar.
        """
        super(EHRPairData, self).__init__()
        self.x_left = x_left
        self.x_right = x_right
        self.edge_index_left = edge_index_left
        self.edge_index_right = edge_index_right
        self.left_edge_type = left_edge_type
        self.right_edge_type = right_edge_type
        self.y = y

    def __inc__(self, key, value):
        if key == 'edge_index_left':
            return self.x_left.size(0)
        if key == 'edge_index_right':
            return self.x_right.size(0)
        else:
            return super(EHRPairData, self).__inc__(key, value)


# construct all the kg relations from the kg relations file
def load_rel(rel_dir):
    # set the threshold for pmi
    pmi_threshold = 1
    # read the relations from the file
    diag_proce_df = pd.read_csv(rel_dir+'diag_proce_rel.csv', sep='\t')
    diag_proce_df = diag_proce_df[diag_proce_df['pmi'] > pmi_threshold]
    diag_pres_df = pd.read_csv(
        rel_dir+'diag_pres_rel.csv', sep='\t', dtype={'tail ent': str})
    diag_pres_df = diag_pres_df[diag_pres_df['pmi'] > pmi_threshold]
    proce_pres_df = pd.read_csv(
        rel_dir+'proce_pres_rel.csv', sep='\t', dtype={'tail ent': str})
    proce_pres_df = proce_pres_df[proce_pres_df['pmi'] > pmi_threshold]

    ndc2rxnorm_file = '../data/ndc_atc/ndc2rxnorm_mapping.txt'
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())

    diag_proce_pairs = []
    diag_proce_df.apply(lambda row: diag_proce_pairs.append(
        str(row['head ent'])+'-'+str(row['tail ent'])), axis=1)

    diag_pres_pairs = []
    diag_pres_df.apply(lambda row: diag_pres_pairs.append(str(
        row['head ent'])+'-'+ndc2rxnorm[row['tail ent']]) if row['tail ent'] in ndc2rxnorm else None, axis=1)

    proce_pres_pairs = []
    proce_pres_df.apply(lambda row: proce_pres_pairs.append(str(
        row['head ent'])+'-'+ndc2rxnorm[row['tail ent']]) if row['tail ent'] in ndc2rxnorm else None, axis=1)

    diag_pres_pairs = set(diag_pres_pairs)
    diag_proce_pairs = set(diag_proce_pairs)
    proce_pres_pairs = set(proce_pres_pairs)

    return diag_proce_pairs, diag_pres_pairs, proce_pres_pairs


# loading the relationship from our knowledge graph
REL_INFOS = load_rel('../data/mimic3/')


def construct_EHR_pairs_dataloader(processed_file, tokenizer, ehr_file, label_file, shuffle=True, batch_size=1, disease_prediction=False):
    """
    Construct EHR pairs data loader

    Args:
    processed_file (str): The file to save the processed data
    tokenizer: The tokenizer to convert the codes to index
    ehr_file (str): The file of EHR data
    label_file (str): The file of label data
    shuffle (bool): Whether to shuffle the data, default is True
    batch_size (int): The batch size, default is 1
    disease_prediction (bool): Whether to predict the disease, True for disease prediction task, False for EHR clustering task

    Returns:
    DataLoader: DataLoader object for loading the data
    """
    if not os.path.exists(processed_file):
        print("ehr infos construct start")
        # Load EHR pairs
        ehr_pairs = load_ehr_pairs(label_file)
        print("ehr pair construct complete")
        ehr_infos = load_ehr_infos(
            ehr_file, tokenizer, disease_prediction=disease_prediction)
        print("ehr infos construct complete")

        data_list = []

        for index in tqdm(range(len(ehr_pairs)),  desc='construct ehr pairs'):
            left_hadm_id, right_hadm_id, label = ehr_pairs[index][
                0], ehr_pairs[index][1], ehr_pairs[index][2]

            # Skip if EHR edge index is empty
            if len(ehr_infos[left_hadm_id].edge_index[0]) == 0:
                continue
            if len(ehr_infos[right_hadm_id].edge_index[0]) == 0:
                continue

            # Convert data to tensor
            left_ehr_x = torch.tensor(
                ehr_infos[left_hadm_id].x, dtype=torch.long).unsqueeze(1)
            right_ehr_x = torch.tensor(
                ehr_infos[right_hadm_id].x, dtype=torch.long).unsqueeze(1)

            left_ehr_edge_index = torch.tensor(
                ehr_infos[left_hadm_id].edge_index, dtype=torch.long)
            right_ehr_edge_index = torch.tensor(
                ehr_infos[right_hadm_id].edge_index, dtype=torch.long)

            left_edge_type = torch.tensor(
                ehr_infos[left_hadm_id].edge_type, dtype=torch.long)
            right_edge_type = torch.tensor(
                ehr_infos[right_hadm_id].edge_type, dtype=torch.long)

            cur_idx_data = EHRPairData(left_ehr_x, left_ehr_edge_index, right_ehr_x, right_ehr_edge_index,
                                       torch.tensor(
                                           label, dtype=torch.float), left_edge_type,
                                       right_edge_type)  # for mse loss
            data_list.append(cur_idx_data)

        if not os.path.exists(os.path.dirname(processed_file)):
            os.makedirs(os.path.dirname(processed_file))
        torch.save(data_list, processed_file)
    else:
        data_list = torch.load(processed_file)
    loader = DataLoader(data_list, batch_size=batch_size,
                        shuffle=shuffle, follow_batch=['x_left', 'x_right'])
    return loader


def construct_dataloder(tokenizer, processed_file, ehr_file, batch_size=1,
                        disease_prediction=False):
    """
    Construct query dataloader for EHR data.

    Args:
    tokenizer: Tokenizer to convert the codes to index.
    processed_file (str): Path to the processed file.
    ehr_file (str): Path to the EHR file.
    batch_size (int): Batch size for dataloader, default is 1.
    disease_prediction (bool): Whether to predict disease, True for disease prediction task, False for EHR clustering task.

    Returns:
    DataLoader: DataLoader object for loading the data.
    List: List of cohorts.
    """

    # Construct query pairs and corresponding diseases
    hadm_ids, diseases = generate_cohort_data(
        ehr_file)

    if not os.path.exists(processed_file):
        ehr_infos = load_ehr_infos(
            ehr_file, tokenizer, disease_prediction=disease_prediction)
        data_list,  cohorts = [], []
        for i in tqdm(range(len(hadm_ids))):
            left_ehr = ehr_infos[hadm_ids[i]]
            left_ehr_x = torch.tensor(
                left_ehr.x, dtype=torch.long).unsqueeze(1)
            left_ehr_edge_index = torch.tensor(
                left_ehr.edge_index, dtype=torch.long)
            left_edge_type = torch.tensor(left_ehr.edge_type, dtype=torch.long)
            cur_idx_data = Data(x=left_ehr_x, edge_index=left_ehr_edge_index,
                                edge_type=left_edge_type)

            data_list.append(cur_idx_data)
            cohorts.append(diseases[i])
            
        if not os.path.exists(os.path.dirname(processed_file)):
            os.makedirs(os.path.dirname(processed_file))
        torch.save((cohorts, data_list), processed_file)
    else:
        cohorts, data_list = torch.load(processed_file)
    loader = DataLoader(data_list, batch_size=batch_size,
                        shuffle=False, follow_batch=['x_left', 'x_right'])
    return loader, cohorts



def load_ehr_infos(ehr_file, tokenizer, disease_prediction=False):
    # Initialize dictionary to store EHR information
    ehr_infos = {}
    ehr_df = pd.read_csv(ehr_file)
    columns_to_convert = ['ATC', 'ICD9_DIAG', 'ICD9_PROCE']
    ehr_df[columns_to_convert] = ehr_df[columns_to_convert].astype(str)

    for _, row in tqdm(ehr_df.iterrows()):

        hadm_id = row['HADM_ID']
        diag_codes = list(set(row['ICD9_DIAG'].split(',')))
        proce_codes = list(set(row['ICD9_PROCE'].split(',')))
        atc_codes = list(set(row['ATC'].split(',')))

        # Create patient graph object using extracted codes and tokenizer
        ehr_graph = UndirectPatientOntoGraphEx(diag_codes, proce_codes, atc_codes, REL_INFOS, tokenizer,
                                               disease_prediction=disease_prediction)

        ehr_infos[hadm_id] = ehr_graph

    return ehr_infos


def load_ehr_pairs(label_file):
    ehr_pairs = []
    label_df = pd.read_csv(label_file, sep='\t')
    for _, row in label_df.iterrows():
        ehr_pairs.append([row[0], row[1], int(row[2])])
    return ehr_pairs


def construct_query_pairs(ehr_file):
    ehr_df = pd.read_csv(ehr_file)
    left_ehr_id = ehr_df.loc[0, 'HADM_ID']
    right_ehr_ids = list(ehr_df.loc[1:, 'HADM_ID'].values)

    return [left_ehr_id]*len(right_ehr_ids), right_ehr_ids


def get_cohorts(test_ehr_file, valid_ehr_file, train_ehr_file):
    train_df, test_df, valid_df = pd.read_csv(train_ehr_file), pd.read_csv(
        test_ehr_file), pd.read_csv(valid_ehr_file)
    diseases = list(set(test_df['disease'].values))
    test_cohorts, valid_cohorts, train_cohorts = [], [], []
    for idx, row in train_df.iterrows():
        train_cohorts.append(diseases.index(row['disease']))
    for idx, row in test_df.iterrows():
        test_cohorts.append(diseases.index(row['disease']))
    for idx, row in valid_df.iterrows():
        valid_cohorts.append(diseases.index(row['disease']))

    return test_cohorts, valid_cohorts, train_cohorts


def generate_cohort_data(ehr_file):
    """
    Generate cohort data from EHR file.

    Args:
        ehr_file (str): Path to the EHR file in CSV format.

    Returns:
        tuple: A tuple containing two lists:
            - hadm_ids (list): List of Hospital Admission IDs (HADM_ID).
            - diseases (list): List of unique diseases extracted from the EHR.

    This function reads the EHR file specified by 'ehr_file' and extracts HADM_IDs
    and diseases associated with each admission. It then generates a list of unique
    diseases and creates a corresponding cohort list based on the index of each
    disease in the unique disease list.

    """

    admission_df = pd.read_csv(ehr_file)
    hadm_ids = admission_df['HADM_ID'].values
    diseases = list(set(admission_df['disease'].values))
    cohorts = []
    for _, row in admission_df.iterrows():
        cohorts.append(diseases.index(row['disease']))

    return hadm_ids, cohorts

# Function to process NDC to ATC mappings
def rxnorm_to_atc_mapping(ndc2atc_file_path):
    ndc2atc_file = open(ndc2atc_file_path, 'r')
    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    rxnorm2atc['RXCUI'] = rxnorm2atc['RXCUI'].map(lambda x: str(x))
    # Create dictionary mapping 'RXCUI' to 'ATC4'
    rxnorm2atc_mapping = rxnorm2atc.set_index('RXCUI')['ATC4'].to_dict()
    return rxnorm2atc_mapping

# Load text embedding from .pkl file
def load_text_embeddings(base_dir):
    # read diagnosis code embeddings
    embs = read_pkl(base_dir + 'diag_desc_emb.pkl')
    diag_embs = {}
    for code, emb in embs.items():
        diag_embs[code[2:].replace('.', '')] = emb

    # read procedure code embeddings
    embs = read_pkl(base_dir + 'proce_desc_emb.pkl')
    proce_embs = {}
    for code, emb in embs.items():
        proce_embs[code[2:].replace('.', '')] = emb

    rxnorm2atc = rxnorm_to_atc_mapping(base_dir + 'ndc2atc_level4.csv')

    # read atc code embeddings
    embs = read_pkl(base_dir + 'atc_desc_emb.pkl')
    atc_embs = {}
    for code, emb in embs.items():
        code = code[2:]
        keys = [key for key, value in rxnorm2atc.items() if value == code]
        for key in keys:
            atc_embs[key] = emb
    return diag_embs, proce_embs, atc_embs


def load_dataset(config, train_processed_file, test_processed_file, valid_cluster_processed_file,
                 test_knn_processed_file, valid_knn_processed_file,
                 train_input_file, batch_size):
    
    # mimic3 or mimic4
    dataset = config.dataset
    base_dir = '../data/'+dataset+'/'
    vocab_file = base_dir+'vocab.pkl'

    tokenizer = EHRTokenizer(vocab_file)

    # read text embeddings for medical codes
    diag_embeddings, proce_embeddings, atc_embeddings = load_text_embeddings(
        '../data/chatgpt_desc/')
    vocab_emb = np.random.randn(len(tokenizer.vocab.word2idx), 768)
    for idx, word in tokenizer.vocab.idx2word.items():
        w_type = word[0]
        if w_type == 'd' and word[2:] in diag_embeddings:
            vocab_emb[idx] = diag_embeddings[word[2:]]
        elif w_type == 'p' and word[2:] in proce_embeddings:
            vocab_emb[idx] = proce_embeddings[word[2:]]
        elif w_type == 'a' and word[2:] in atc_embeddings:
            vocab_emb[idx] = atc_embeddings[word[2:]]
    vocab_emb = torch.tensor(vocab_emb, dtype=torch.float)

    train_ehr_file, train_label_file = base_dir + \
        'train_admissions.csv', base_dir + 'train_label.csv'
    valid_ehr_file, valid_label_file = base_dir + \
        'valid_admissions.csv', base_dir + 'valid_label.csv'
    test_ehr_file = base_dir + 'test_admissions.csv'

    # True for disease prediction task, False for EHR clustering task
    disease_prediction = False
    if config.task == 'knn':
        disease_prediction = True

    # construct dataloader for EHR pairs
    train_dataloader = construct_EHR_pairs_dataloader(train_processed_file, tokenizer, train_ehr_file, train_label_file,
                                                      batch_size=batch_size, disease_prediction=disease_prediction)

    test_dataloader, _ = construct_dataloder(tokenizer, test_processed_file, test_ehr_file, batch_size=batch_size,
                                             disease_prediction=False)
    valid_cluster_dataloader, _ = construct_dataloder(tokenizer, valid_cluster_processed_file, valid_ehr_file, batch_size=batch_size,
                                                      disease_prediction=False)

    valid_knn_dataloader, _ = construct_dataloder(tokenizer, valid_knn_processed_file, valid_ehr_file, batch_size=batch_size,
                                                  disease_prediction=True)
    test_knn_dataloader, _ = construct_dataloder(tokenizer, test_knn_processed_file, test_ehr_file, batch_size=batch_size,
                                                 disease_prediction=True)
    train_input_dataloader, _ = construct_dataloder(tokenizer, train_input_file, train_ehr_file, batch_size=batch_size,
                                                    disease_prediction=True)

    test_cohorts, valid_cohorts, train_cohorts = get_cohorts(
        test_ehr_file, valid_ehr_file, train_ehr_file)

    return vocab_emb, tokenizer, train_dataloader, test_dataloader, valid_cluster_dataloader, test_knn_dataloader, valid_knn_dataloader, \
        test_cohorts, valid_cohorts, train_input_dataloader, train_cohorts
