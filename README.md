# SEFP: structure-based enzyme function 
prediction
![image](https://github.com/PrestigeOfGod/SEFP/assets/39855922/0e7baca5-e0bf-4520-96f0-4832b5dfa283)
Overview of SEFP: SEFP processes enzyme point clouds extracted from PDB files as its input. These point clouds consist of Cα coordinates and integer representations of each amino acid residue. The framework comprises two main encoders: the Point Cloud-Based Enzyme Structure Encoder, which processes the coordinates, and the Bio-BCS Residue Feature Encoder, which handles the amino acid residue representations. The outputs from these encoders are fused by a Multi-Layer Perceptron (MLP) to generate the final EC number prediction.
![image](https://github.com/PrestigeOfGod/SEFP/assets/39855922/4deeca82-87ed-4c8d-9833-ebe23ac327c2)
The detailed illustration of point cloud-based enzyme structure encoder. The Residue feature adapter is the core component for the interaction between the Residue Global attention module and the tailored enzyme point cloud network. It generates multi-scale point feature by merge the residue feature from a residue global attention block and point feature from the previous set abstraction unit, and input it to its corresponding set abstraction unit.

requirements:
pytorch, transformers

Adjust parameters of the train_sefp.py

model training:

python train_sefp.py
