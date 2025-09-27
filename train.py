from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import trange
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import ydf
import xgboost as xgb


def feature_extraction(df: pd.DataFrame, radius: int, fp_size: int, drop_cols: list = None) -> pd.DataFrame:
    atom_num, atom_bonds_num, num_hba, num_hbd, tpsa, hdonors, haacceptors, valence_electrons, radical_electrons, mol_wt, mol_log_p, mol_mr, rot_bonds, fraction_csp3, num_rings, num_aromatic_rings, morgan_fingerprint = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for i in trange(len(df)):
        row = df["SMILES"].iloc[i]
        mol = Chem.MolFromSmiles(row)

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
        morgan_fp = gen.GetFingerprint(mol)
        arr = np.zeros((1024,), dtype=int)
        ConvertToNumpyArray(morgan_fp, arr)
        morgan_fingerprint.append(arr)

        atom_num.append(mol.GetNumAtoms())
        atom_bonds_num.append(mol.GetNumBonds())

        descriptors = Descriptors.CalcMolDescriptors(mol)
        tpsa.append(descriptors["TPSA"])
        hdonors.append(descriptors["NumHDonors"])
        valence_electrons.append(descriptors["NumValenceElectrons"])
        radical_electrons.append(descriptors["NumRadicalElectrons"])
        mol_wt.append(descriptors["MolWt"])
        mol_log_p.append(descriptors["MolLogP"])
        mol_mr.append(descriptors["MolMR"])

        # oxidation_numbers.append(rdMolDescriptors.CalcOxidationNumbers(mol))
        num_hbd.append(rdMolDescriptors.CalcNumHBD(mol))
        num_hba.append(rdMolDescriptors.CalcNumHBA(mol))
        haacceptors.append(rdMolDescriptors.CalcNumHBA(mol))
        rot_bonds.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        fraction_csp3.append(rdMolDescriptors.CalcFractionCSP3(mol))
        num_rings.append(rdMolDescriptors.CalcNumRings(mol))
        num_aromatic_rings.append(rdMolDescriptors.CalcNumAromaticRings(mol))

    df["ATOM_NUM"] = atom_num
    df["ATOM_BONDS_NUM"] = atom_bonds_num
    df["NUM_HBA"] = num_hba
    df["NUM_HBD"] = num_hbd
    df["TPSA"] = tpsa
    df["HDONORS"] = hdonors
    df["VALENCE_ELECTRONS"] = valence_electrons
    df["RADICAL_ELECTRONS"] = radical_electrons
    df["MOL_WT"] = mol_wt
    df["MOL_LOG_P"] = mol_log_p
    df["MOL_MR"] = mol_mr
    df["ROT_BONDS"] = rot_bonds
    df["FRACTION_CSP3"] = fraction_csp3
    df["NUM_RINGS"] = num_rings
    df["NUM_AROMATIC_RINGS"] = num_aromatic_rings
    # df["OXIDATION_NUMBERS"] = oxidation_numbers

    morgan_fp = pd.DataFrame(morgan_fingerprint, columns=[f"morgan_{i}" for i in range(fp_size)])
    df = pd.concat([df.reset_index(drop=True), morgan_fp], axis=1)

    if drop_cols is not None:
        df.drop(columns=drop_cols, axis=1, inplace=True)
        return df
    else:
        return df

def gradient_boosted_trees_learner():
    df_train = pd.read_csv('melting-point/train.csv')
    df_valid = pd.read_csv('melting-point/test.csv')

    # feature extraction
    df_train = feature_extraction(df_train, radius=2, fp_size=1024, drop_cols=["SMILES", "id"])
    df_valid = feature_extraction(df_valid, radius=2, fp_size=1024, drop_cols=["SMILES", "id"])
    print(df_train.head())

    input_features = df_train.columns.tolist()
    input_features.remove("Tm")
    gb_forest_model = ydf.GradientBoostedTreesLearner(label="Tm",
                                                      features=input_features,
                                                      task=ydf.Task.REGRESSION,
                                                      loss="SQUARED_ERROR",
                                                      max_depth=512,
                                                      num_trees=1000,
                                                      max_num_nodes=-1,
                                                      growing_strategy="BEST_FIRST_GLOBAL",
                                                      num_threads=256,
                                                      sampling_method="RANDOM",
                                                      l1_regularization=0.1,
                                                      l2_regularization=1.0,
                                                      ).train(df_train, verbose=2)

    preds = gb_forest_model.predict(df_valid)
    print("# predictions: ", preds[:15])
    sample_submission = pd.read_csv("melting-point/sample_submission_v1.csv")
    sample_submission["Tm"] = preds

    sample_submission.to_csv("melting-point/sample_submission_v3.csv", index=False)
    gb_forest_model.save("gb_forest_model_v3")

def xgboost():
    df_train = pd.read_csv('melting-point/train.csv')
    df_valid = pd.read_csv('melting-point/test.csv')

    # feature extraction
    X = feature_extraction(df_train, radius=2, fp_size=1024, drop_cols=["SMILES", "id"])
    y = X["Tm"].values
    X = X.drop(columns=["Tm"])
    X_valid = feature_extraction(df_valid, radius=2, fp_size=1024, drop_cols=["SMILES", "id"])

    X_train, X_test, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_valid)
    dtest = xgb.DMatrix(X_valid)

    param = {
        "max_depth": 16,
        "eta": 0.2,
        "gamma": 0.1,
        "max_leaves": 256,
        "objective": "reg:squarederror",
        "eval_metric": "mae"
        }
    evallist = [(dtrain, "train"), (dvalid, "eval")]

    num_round = 100
    bst = xgb.train(param, dtrain, num_round, evals=evallist)

    preds = bst.predict(dtest)
    print(preds[:15])
    sample_submission = pd.read_csv("melting-point/sample_submission.csv")
    sample_submission["Tm"] = preds

    sample_submission.to_csv("melting-point/sample_submission_v3.csv", index=False)
    bst.save_model('xgboost/xgb_model_v1_model.ubj')

if __name__ == '__main__':
    #gradient_boosted_trees_learner()
    xgboost()