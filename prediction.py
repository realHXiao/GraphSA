import os
import sys
import json
import paddle
import numpy as np
import pandas as pd
import pickle
import argparse

from rdkit import Chem
from rdkit.Chem import AllChem
from pahelix.utils.compound_tools import mol_to_graph_data
from src.data_gen import DataCollateFunc
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from pahelix.utils.data_utils import load_npz_to_data_list
import matplotlib.pyplot as plt

def from_csv_and_csv_file(args): 
    SA_data = pd.read_csv(os.path.join(args.csv_file))
    n = 0

    data_seq = []
    seq_df = pd.read_csv(os.path.join(args.input_seq), sep = "\t")
    X_seq = seq_df.values
    for i in range(len(X_seq)):
        protein_ = X_seq[i,1:]
        data_seq.append(protein_)
    SA_data['protein_seq'] = data_seq

    data_lst = []
    for ind in range(len(SA_data.values)) : 
        protein_seq = SA_data['protein_seq'][ind]
        ligand = SA_data['Substrate_SMILES'][ind]
        
        if "." not in ligand:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(ligand),
                            isomericSmiles=True)
            mol = AllChem.MolFromSmiles(smiles)
            mol_graph = mol_to_graph_data(mol)
            data0 = {s: v for s, v in mol_graph.items()}
            data0['protein_seq'] = np.array([protein_seq])

            data_lst.append(data0) 

    return data_lst    

def prediction(args, pred_data, model_config, model):
    compound_protein = []
    for data in pred_data:
        infer_collate_fn = DataCollateFunc(
            model_config['compound']['atom_names'],
            model_config['compound']['bond_names'],
            is_inference=True,
            label_name=args.label_type)

        graphs, proteins_seq = infer_collate_fn([data])
        graphs = graphs.tensor()
        proteins_seq = paddle.to_tensor(proteins_seq)

        model.eval()
        compound_protein_ = model(graphs, proteins_seq)
        compound_protein.append(compound_protein_.tolist())

    return compound_protein

def input_csv_output_csv_file(args, pred):
    SA_data = pd.read_csv(os.path.join(args.csv_file))
    prediction_df = pd.DataFrame(columns= ["Protein_sequence", "Substrate_SMILES","SA_pred"])
    SA_pred = 10 ** pred
    protein_sequence, substrate = [], []
    for ind in range(len(SA_data.values)):
        ligand = SA_data['Substrate_SMILES'][ind]
        protein = SA_data['Protein_sequence'][ind]
        if "." not in ligand:
            protein_sequence.append(protein)
            substrate.append(ligand) 
    prediction_df['Protein_sequence'] = protein_sequence
    prediction_df['Substrate_SMILES'] = substrate

    value_name = '{}_pred'.format(args.label_type)
    prediction_df[value_name] = SA_pred
    prediction_df = prediction_df.sort_values(axis=0, by=value_name, ascending=False)
    return prediction_df

def from_fasta_and_csv_file(args, model_config, model): 
    ligand = open(os.path.join(args.SMILES_file), 'r')
    ligand = ligand.read()
    seq_df = pd.read_csv(os.path.join(args.input_seq), sep = "\t")

    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(ligand),
                isomericSmiles=True)
    mol = AllChem.MolFromSmiles(smiles)
    mol_graph = mol_to_graph_data(mol)

    X_seq = seq_df.values
    compound_protein = []
    for i in range(len(X_seq)):
        protein_seq = X_seq[i,1:]

        data = {k: v for k, v in mol_graph.items()}
        data['protein_seq'] = np.array(protein_seq)

        infer_collate_fn = DataCollateFunc(
            model_config['compound']['atom_names'],
            model_config['compound']['bond_names'],
            is_inference=True,
            label_name=args.label_type)

        graphs, proteins_seq = infer_collate_fn([data])
        graphs = graphs.tensor()
        proteins_seq = paddle.to_tensor(proteins_seq)

        model.eval()
        pred = model(graphs, proteins_seq)
        
    return pred

def input_fasta_output_csv_file(args, pred):
    protein_seq = open(os.path.join(args.fasta_file), 'r')
    lines = protein_seq.readlines()
    organism = []
    sequence, sequences = [], []
    n = 0
    for i in lines:
        n += 1
        line = i.strip('\n').split()
        word = list(line[0])
        if word[0] == '>':
            organism.append(i.strip('\n'))
        if word[0] != '>' and n > 1:
            sequence.append(line[0])
        else:
            seq = ''.join(sequence)
    #         print(seq)
            sequences.append(seq)
            sequence.clear()
            
    seq = ''.join(sequence)
    sequences.append(seq)

    prediction_df = pd.DataFrame(columns= ["Organism", "Protein_sequence", "Substrate_SMILES", "SA_pred"])
    prediction_df['Organism'] = organism
    prediction_df['Protein_sequence'] = sequences[1:]
    prediction_df['Substrate_SMILES'] = open(os.path.join(args.SMILES_file), 'r').read()

    pred_value = 10 ** pred
    value_name = '{}_pred'.format(args.label_type)
    prediction_df[value_name] = pred_value
    prediction_df = prediction_df.sort_values(axis=0, by=value_name, ascending=False)
    return prediction_df

def main(args): 
    model_config = json.load(open(args.model_config, 'r'))

    # load model parameters. 
    model = SAModel(model_config)
    model.set_state_dict(paddle.load(os.path.join(args.model_file)))
    
    if args.csv_file_input: 
        pred_data = from_csv_and_csv_file(args)
        # predict
        pred = prediction(args, pred_data, model_config, model)
    
    if args.fasta_file_input:    
        # protein sequences fasta file with one substrate SMILES code. 
        pred = from_fasta_and_csv_file(args, model_config, model)
    
    # save results in csv file.
    if args.csv_file_input: 
        prediction_df = input_csv_output_csv_file(args, pred)
        prediction_df.to_csv(args.output_csv_file)
    
    # save results in csv file.
    if args.fasta_file_input:
        prediction_df = input_fasta_output_csv_file(args, pred)
        prediction_df.to_csv(args.output_csv_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label_type', type=str, default='SA', help = 'Optional! Default SA. choose SA, KM or kcat.')
    parser.add_argument("--csv_file", type=str, help = 'load csv file containing protein sequences, substrate SMILES codes. Required after -c or --csv_file_input is specified.')
    parser.add_argument("-c", "--csv_file_input", action="store_true", default=False, help = 'Optional! Default False (Disable). ')
    parser.add_argument("--input_seq", type=str, help = 'Required! Load csv file of protein sequence vectors.')
    parser.add_argument("--model_config", type=str, help = 'Required!')
    parser.add_argument("-m", "--model_file", type=str, help = 'Required! Load model which is trained by train.py.')
    parser.add_argument("--fasta_file", type=str, help = 'load fasta file from BLAST. Required after -f or --fasta_file_input is specified.')
    parser.add_argument("-f", "--fasta_file_input", action="store_true", default=False, help = 'Optional! Default False (Disable). ')
    parser.add_argument("-S", "--SMILES_file", type=str, help = "Required! load txt file containing one type SMILES code.")
    parser.add_argument("-o", "--output_csv_file", type=str, default='prediction.csv', help = 'Optional! ')
    args = parser.parse_args()
    main(args)
