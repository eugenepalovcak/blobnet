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

from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from prody import parsePDB

from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

import glob

def load_protein_structure(path):
    """ Load a protein structure and return coords as Nx3 array """
    pdb = parsePDB(path).select('name CA CB')
    coords = pdb.getCoords()
    coords -= coords.mean(axis=0)
    return coords

class ProteinDataset(Dataset):
    """ Makes a pytorch-geometric dataset of point clouds from a 
    directory of PDB files """

    def __init__(self, pdb_dir, transform=None, pre_transform=None):
        super(ProteinDataset, self).__init__(pdb_dir, transform, pre_transform)
        self.pdb_files = glob.glob(pdb_dir + "/*.pdb")

    def __len__(self):
        return len(self.pdb_files)

    def _download(self):
        pass
  
    def _process(self):
        pass

    def get(self, idx):
        pdb_file = self.pdb_files[idx]
        coords = load_protein_structure(pdb_file)
        data = Data(x=coords)
        return data





if __name__=="__main__":
    D = ProteinDataset("./dompdb")
    loader = DataLoader(D, batch_size=16)
    for i in loader:
       print(i)
