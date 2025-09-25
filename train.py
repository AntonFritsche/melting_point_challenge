from model import lstm_embedding, lstm_stacked
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import trange

import pandas as pd
import tensorflow as tf


def main():
    df = pd.read_csv('melting-point/train.csv')
    atom_num = []
    atom_bonds_num = []
    tpsa = []
    hdonors = []
    valence_electrons = []
    radical_electrons = []

    for i in trange(len(df)):
        row = df["SMILES"].iloc[i]
        mol = Chem.MolFromSmiles(row)
        atom_num.append(mol.GetNumAtoms())
        atom_bonds_num.append(mol.GetNumBonds())
        descriptors = Descriptors.CalcMolDescriptors(mol)
        tpsa.append(descriptors["TPSA"])
        hdonors.append(descriptors["NumHDonors"])
        valence_electrons.append(descriptors["NumValenceElectrons"])
        radical_electrons.append(descriptors["NumRadicalElectrons"])

    df["ATOM_NUM"] = atom_num
    df["ATOM_BONDS_NUM"] = atom_bonds_num
    df["TPSA"] = tpsa
    df["HDONORS"] = hdonors
    df["VALENCE_ELECTRONS"] = valence_electrons
    df["RADICAL_ELECTRONS"] = radical_electrons

    y = df["Tm"].values
    df.drop(columns=["SMILES", "Tm", "id"], inplace=True)
    X = df.values

    print(df[:3])

    embedding_size = 1024
    model_embedding = lstm_embedding(len(df.columns), embedding_size, )
    model_embedding.summary()

    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01)
    model_embedding.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsoluteError, metrics=['accuracy'])
    model_embedding.fit(X, y, epochs=25, batch_size=512, verbose=2)

    #model_stacked = lstm_stacked(len(features))
    #model_stacked.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model_stacked.fit(X, y, epochs=10, batch_size=256, verbose=1)

if __name__ == '__main__':
    main()