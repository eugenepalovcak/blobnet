#! /usr/bin/env python
# Copyright (C) 2018 Eugene Palovcak
# University of California, San Francisco
#
# Data-loader for pushing protein structure clouds 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
from prody import parsePDB
import os

def pdb2pc(pdb_file):
    """ Loads a PDB file and outputs a point cloud tensor """
    pdb = parsePDB(pdb_file).select('name CA CB')
    coords = pdb.getCoords()
    coords -= coords.mean(axis=0)
    return Data(x=torch.tensor(coords))


class ProteinDataset(Dataset):
    """ Makes a dataset of unlabeled point clouds from a list of PDB files """
    def __init__(self, pdb_files):
        super(ProteinDataset).__init__()
        self.pdb_files = pdb_files

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        """ Replace this line with whatever preprocessing or
        data-loading that is necessary """
        return pdb2pc(self.pdb_files[idx])

def get_loaders(triplet_file, pdb_dir, batch_size):
    """ Gets anchor, positive, and negative data loader """
    
    with open(triplet_file, "r") as f:
        triplets = [[os.path.join(pdb_dir, pdb)
                     for pdb in line.strip().split(",")]
                     for line in f.readlines()]
    
    anchors, positives, negatives = [DataLoader(ProteinDataset(list(t)),
                                     batch_size=batch_size)
                                     for t in zip(*triplets)]

    return anchors, positives, negatives


if __name__=="__main__":
    """ Basic method for iterating over training triplets """
    triplet_file = "triplets_noise1.csv" 
    pdb_dir = "./dompdb"
    batch_size = 8

    anch, pos, neg = get_loaders(triplet_file, pdb_dir, batch_size) 

    for a,p,n in zip(anch,pos,neg):
        print(a,p,n)

