import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.ML.Cluster import Butina
import pandas as pd
import random
RANDOM_SEED = 42


def smiles_to_ecfp(smiles: list, radius: int = 2, nbits: int = 1024):
    """Calculate the morgan fingerprint of a list of smiles. Return a numpy array"""

    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    return np.asarray(fps)


def cluster_smiles(smiles: list, clustering_cutoff: float = 0.4):
    """ Cluster smiles based on their Murcko scaffold using the Butina algorithm:

    D Butina 'Unsupervised Database Clustering Based on Daylight's Fingerprint and Tanimoto Similarity:
    A Fast and Automated Way to Cluster Small and Large Data Sets', JCICS, 39, 747-750 (1999)
    """
    # Make Murcko scaffolds
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    mols = [GetScaffoldForMol(m) for m in mols]

    # Create fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]

    # Cluster fingerprints
    clusters = clusterfp(fps, clustering_cutoff=clustering_cutoff)

    return clusters


def clusterfp(fps, clustering_cutoff: float = 0.4):
    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, clustering_cutoff, isDistData=True)
    return cs


def split_smiles(smiles: list, test_split: float = 0.2, val_split: float = 0.1):
    train, test, val = [], [], []
    random.seed(RANDOM_SEED)

    clusters = cluster_smiles(smiles, clustering_cutoff=0.3)

    for clust in clusters:
        for idx in clust:
            random_nr = random.uniform(0, 1)
            if random_nr < val_split:
                val.append(idx)
            elif val_split < random_nr < (test_split + val_split):
                test.append(idx)
            else:
                train.append(idx)

    return train, test, val


def prep_data():

    train_data = pd.read_csv('data/CHEMBL2047_EC50_train.csv')
    test_data = pd.read_csv('data/CHEMBL2047_EC50_test.csv')

    train_idx, val_idx, _ = split_smiles(train_data['smiles'], test_split=0.1, val_split=0)
    x_train = smiles_to_ecfp(train_data['smiles'][train_idx])
    y_train = np.array([-np.log10(i) for i in train_data['exp_mean [nM]'][train_idx]])

    x_val = smiles_to_ecfp(train_data['smiles'][val_idx])
    y_val = np.array([-np.log10(i) for i in train_data['exp_mean [nM]'][val_idx]])

    x_test = smiles_to_ecfp(test_data['smiles'])
    y_test = np.array([-np.log10(i) for i in test_data['exp_mean [nM]']])

    return x_train, y_train, x_val, y_val, x_test, y_test



