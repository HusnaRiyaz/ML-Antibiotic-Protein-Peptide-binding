# Mount your drive
from google.colab import drive
drive.mount('/content/drive')

# install biopython
pip install biopython

# import the required packages
import re
import json
import math as m
from Bio.PDB.PDBParser import PDBParser
import pandas as pd
import itertools 
import numpy as np
import re

#Reading all file name from main file to a list
files=[]
with open("/content/drive/MyDrive/Antibio/Final_antibio_pdb_id.txt",'r') as file:
  for line in file.readlines():
    row=line.split()
    #print(row)
    if len(row)>=0:
      each_file=row[0]
      #print(each_file)
      

      filename="/content/drive/MyDrive/Antibio/complex/" +each_file
      files.append(filename) 
      #print(files)

#Function to write output to text file
def write(result,filename):
  # open file in write mode
  with open(filename, 'w') as fp:
      for item in result:
          # write each item on a new line
          fp.write("%s\n" % item)

#Method to get only those residues that are within the specified distance (4 Å) from the peptides
def processor(id,path,write_filename):
  write_data=[]
  parser = PDBParser()
  structure = parser.get_structure(id,path)
  #structure = parser.get_structure('1d8t',"/content/drive/MyDrive/Antibio/complex/1d8t_C_A.pdb")

  atoms = structure.get_atoms()
  chain_list = []
  chain1 = []
  chain2 = []
  #result = []
  #d={chain1:[],chain2:[]}
  for atom in atoms:
    atom_name = (atom.get_name())
    atom_id = (int(str(atom.get_serial_number()).strip()))
    residue_name = (atom.get_parent().get_resname().strip())
    residue_id = (int(str(atom.get_parent().get_id()[1]).strip()))
    x,y,z = (float(atom.get_coord()[0]),float(atom.get_coord()[1]),float(atom.get_coord()[2]))
    chain_id = (atom.get_parent().get_parent().get_id())
    val = [atom_name, atom_id,residue_id, residue_name, chain_id,x,y,z]
    #print(val)
    if chain_id not in chain_list:
      chain_list.append(chain_id)
      
    if chain_id == chain_list[0]:
      chain1.append(val)
    #print(chain1)
    elif chain_id == chain_list[1]:
      chain2.append(val)

  for i in chain1 :
    x1 = i[5]
    y1 = i[6]
    z1 = i[7]
    atm_id1 = i[1]
    atm_name1 = i[0]
    res_name1 = i[3]
    res_id1 = i[2]
    chain_id1 = i[4]
    for j in chain2:
      atm_id2 = j[1]
      atm_name2 = j[0]
      x2 = j[5]  
      y2 = j[6]
      z2 = j[7]
      res_name2 = j[3]
      res_id2 = j[2]
      chain_id2 = j[4]
      #print(x1,y1,z1,x2,y2,z2)
      dist = ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
      if float(dist) <= 4.0:
        result =("("+res_name1+", "+str(res_id1)+", "+atm_name1+", "+str(chain_id1)+") - ("+res_name2+", "+str(res_id2)+", "+atm_name2+", "+str(chain_id2)+") "+str(dist))
        write_data.append(result)
  write(write_data,write_filename)
  print(write_data)


#Iterating over multiple files and invoking method
for each_file in files:
  path=each_file+".pdb"
  file_name=each_file.split('/')[-1]
  target_file="/content/drive/MyDrive/Antibio/final_res/"+each_file.split('/')[-1]+".txt"
  id=file_name.split('_')[0]
  print(path)
  processor(id,path,target_file)
  

#To get the CA details of the residues
#def CA_details (file, path, id) :
def CA_details(file):
  CA = []

  for line in file :
    line = line.split(',')
    res_id1 = line[1] 
    parser = PDBParser()
    #structure = parser.get_structure('1d8t', "/content/drive/MyDrive/Antibio/complex/1d8t_D_B.pdb")

    structure = parser.get_structure(id, path )
    atoms = structure.get_atoms()
    for atom in atoms:
      CA_PDB_residue_id = (int(str(atom.get_parent().get_id()[1]).strip()))
      CA_residue_name = (atom.get_parent().get_resname().strip())
      chain_id = (atom.get_parent().get_parent().get_id())
      #grp = get_group(residue_name)
      # res_name - centroid
      if int(res_id1) == int(CA_PDB_residue_id) :
        atom_name = (atom.get_name())
        if atom_name == 'CA'  :
          x,y,z = (float(atom.get_coord()[0]),float(atom.get_coord()[1]),float(atom.get_coord()[2]))
          val = [atom_name, CA_residue_name, CA_PDB_residue_id, x,y,z]
          if val not in CA :
            CA.append([atom_name, CA_residue_name, CA_PDB_residue_id,  chain_id, x,y,z])

  return CA


# create a dictionary for groups of amino acids into their specific residue groups
resname_encoding = {'ALA':0, 'MET': 0, 'VAL':0,'ILE':0,'LEU':0,'GLY':0,'PRO':0,'MLE':0, #Parent Leucine(LEU)
                 'MLU':0, #Parent Leucine(LEU)
                 'GHP':0, #Parent Glycine(GLY)
                 'DAL':0, # Parent Alanine(ALA)
                 'MVA':0, #Parent Valine(VAL)
                 'LYS':1,'ARG':1,'HIS':1,
                 'ASP':2,'GLU':2,'GLN':2,'ASN':2,
                 'TYR':3,'PHE':3,'TRP':3,
                 'CYS':4,'SER':4,'THR':4,
                 'OTZ':4, # Parent CYS,SER
                 'CCS':4, # Parent CYSTEINE(CYS)
                 'SAH':4, # Parent CYSTEINE(CYS)
                 'DSN':4, # Parent Serine(SER)
                 'FMN':4, # Parent Threonine(THR)
                 'B3T':5,'CL':5, 'IOD':5, 'NA':5, 'F7P':5,'XCP':5, 'B3E':5, 'NH2':5, 'ZN':5, 'CD':5,'BGC':5}

#ignore_res_names = [ 'XPC','B3E','NH2','ZN','CD','B3T']

#Iterating over multiple files for CA
for every_file in files:
  target_filename = every_file + '.pdb'
  print(target_filename)
  target_pdb_id = every_file.split("/") [-1].split("_")[0]
  files_new = open('/content/drive/MyDrive/Antibio/final_res/' +every_file.split("/") [-1] +'.txt','r')
  #files_new = open("/content/drive/MyDrive/Antibio/final_res/1d8t_C_A.txt",'r')
  files_new = files_new.readlines()
  CA_results = CA_details(files_new) 
  #print(target_filename)
  final=[]
  for i in CA_results:
    i.append(resname_encoding[i[2]])
    if i not in final:
      final.append(i)
  for i in final:
    print(i)
  
# To get the CB details of the residues
def CB_details (file,path,id) :
  CB = []
  for line in file :
    line = line.split(',')
    res_id1 = line[1] 
    parser = PDBParser()
    #structure = parser.get_structure('1d8t',"/content/drive/MyDrive/Antibio/complex/"+"1d8t_D_B.pdb")
    structure = parser.get_structure(id, path )
    atoms = structure.get_atoms()
    for atom in atoms:
      CB_PDB_residue_id = (int(str(atom.get_parent().get_id()[1]).strip()))
      CB_residue_name = (atom.get_parent().get_resname().strip())
      chain_id = (atom.get_parent().get_parent().get_id())
      #grp = get_group(residue_name)
      # res_name - centroid
      if int(res_id1) == int(CB_PDB_residue_id) :
        atom_name = (atom.get_name())
        if atom_name == 'CB'  :
          x,y,z = (float(atom.get_coord()[0]),float(atom.get_coord()[1]),float(atom.get_coord()[2]))
          val = [atom_name, CB_residue_name,CB_PDB_residue_id, x,y,z]
          if val not in CB :
            CB.append([atom_name, CB_residue_name, CB_PDB_residue_id,chain_id, x,y,z])

  return CB

#Iterating over multiple files for CB
for every_file in files:
  target_filename = every_file + '.pdb'
  target_pdb_id = every_file.split("/") [-1].split("_")[0]
  files_new = open('/content/drive/MyDrive/Antibio/final_res/' +every_file.split("/") [-1] +'.txt','r')
  files_new = files_new.readlines()
  CB_results = CB_details(files_new, target_filename, target_pdb_id) 
  print(target_filename)
  final=[]
  for i in CB_results:
    i.append(resname_encoding[i[1]])
    if i not in final:
      final.append(i)
  for i in final:
    print(i)


# All those atoms side chain residues except CA & CB retrieval
def Centroid_details (file,path,id) :
  Centroid = []
  x_cords = {}
  y_cords = {}
  z_cords = {}
  ignore_atoms = ['N','C','O','OXT', 'CA','CB', 'CL', 'NA', 'IOD', 'F7P']
  ignore_res = ['ALA', 'GLY']
  d={}
  for line in file :
    #print(line)
    line = line.split(',')
    res_id1 = line[1]
    if (line[0],line[1]) not in d.keys():
      d[(line[0],line[1])]=1
      #print((line[0][1:],int(line[1])))
      key=(line[0][1:],int(line[1]))
      #x_cords[key]=[]
      #y_cords[key]=[]
      #z_cords[key]=[]
      #print("Text",res_id1)
      parser = PDBParser()
      #structure = parser.get_structure('1d8t', "/content/drive/MyDrive/Antibio/complex/1d8t_C_A.pdb")
      structure = parser.get_structure(id, path )
      atoms = structure.get_atoms()
      for atom in atoms:
        Cent_atom_name = (atom.get_name())
        Cent_PDB_residue_id = (int(str(atom.get_parent().get_id()[1]).strip()))
        Cent_residue_name = (atom.get_parent().get_resname().strip())
        Cent_chain_id = (atom.get_parent().get_parent().get_id())
        #print(Cent_atom_name,Cent_PDB_residue_id,Cent_residue_name)
        #grp = get_group(residue_name)
        # res_name - centroid
        if int(res_id1) == int(Cent_PDB_residue_id) :
          #print("Id is ---------",int(res_id1))
          #print(float(atom.get_coord()[0]),float(atom.get_coord()[1]),float(atom.get_coord()[2]))
          if Cent_residue_name not in ignore_res and Cent_atom_name not in ignore_atoms:
          
              atom_name = (atom.get_name())
              if atom_name != 'CA' or 'CB'  :
                x,y,z = (float(atom.get_coord()[0]),float(atom.get_coord()[1]),float(atom.get_coord()[2]))
                if (Cent_residue_name,Cent_PDB_residue_id,Cent_chain_id) not in x_cords.keys():
                   x_cords[(Cent_residue_name,Cent_PDB_residue_id,Cent_chain_id)] = []
                if (Cent_residue_name,Cent_PDB_residue_id,Cent_chain_id) not in y_cords.keys():
                   y_cords[(Cent_residue_name,Cent_PDB_residue_id,Cent_chain_id)] = []
                if (Cent_residue_name,Cent_PDB_residue_id,Cent_chain_id) not in z_cords.keys():
                   z_cords[(Cent_residue_name,Cent_PDB_residue_id,Cent_chain_id)] = []
                x_cords[(Cent_residue_name,Cent_PDB_residue_id,Cent_chain_id)].append(x)
                y_cords[(Cent_residue_name,Cent_PDB_residue_id,Cent_chain_id)].append(y)
                z_cords[(Cent_residue_name,Cent_PDB_residue_id,Cent_chain_id)].append(z)


                val = [atom_name, Cent_residue_name, Cent_PDB_residue_id,  Cent_chain_id]
                
                Centroid.append([atom_name, Cent_residue_name, Cent_PDB_residue_id,  Cent_chain_id])
  
  return (x_cords,y_cords,z_cords)
  # ['CB', 23, 'ARG', 'K', 5.7210001945495605, 71.1989974975586, -64.09700012207031, 1]


# To retrieve each files CT details for multiple files
import numpy as np
for every_file in files:
  target_filename = every_file + '.pdb'
  print(target_filename)
  final_centroid_all = []
  target_pdb_id = every_file.split("/") [-1].split("_")[0]
  files_new = open('/content/drive/MyDrive/Antibio/final_res/' +every_file.split("/") [-1] +'.txt','r')
#files = open("/content/drive/MyDrive/Antibio/res/1d8t_C_A.txt",'r')
  files_new = files_new.readlines()
  x,y,z = Centroid_details(files_new, target_filename, target_pdb_id)
  for each_pair in x.keys():
    centroid_all = ['CT' , each_pair[0],each_pair[1],each_pair[2],np.mean(x[each_pair]),np.mean(y[each_pair]),np.mean(z[each_pair])]
    final_centroid_all.append(centroid_all)
    #print(final_centroid_all)
  final=[]
  for i in final_centroid_all :
    #print(i)
    i.append(resname_encoding[i[1]])
    if i not in final:
      final.append(i)
  for i in final:
        print(i)


# Distance calculation within specific pockets
import csv
def writecsv(rows,file):
  #field = ['atm_name1','res_name1','res_id1','chain_id1','group_id1',"-",'atm_name2','res_name2','res_id2','chain_id2','group_id2',"----",'dist']
  #field=['c1','c2','c3','c4','x','y','z','tar']
  with open(file, 'w') as f:
        
      # using csv.writer method from CSV package
      write = csv.writer(f)
        
      #write.writerow(field)
      write.writerows(rows)
files=[]
#with open("/content/drive/MyDrive/Antibio/Final_antibio_pdb_id.txt",'r') as file:
with open("/content/drive/MyDrive/Antibio/Negative_pdb_id.txt",'r') as file:
  for line in file.readlines():
    row=line.split()
    #print(row)
    if len(row)>=0:
      each_file=row[0]
      #print(each_file)
      

      filename="/content/drive/MyDrive/Antibio/Negative_CA_CB_CT/" +each_file
      #filename="/content/drive/MyDrive/Antibio/Binning_CA_CB_CT_positive/" +each_file
      files.append(filename) 
      #print(files)

# CA with CA
import csv
def binning(id,path,write_filename):
    ca=[]
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            ca.append(row)
    ca=ca[1:]
    final_op =[]
    for i in range(len(ca)-1):
      
        #print(ca[i],type(ca[i]))
        atm_name1=ca[i][0]
        res_id1=ca[i][1]
        res_name1=ca[i][2]
        chain_id1=ca[i][3]
        x1=float(ca[i][4])
        y1=float(ca[i][5])
        z1=float(ca[i][6])
        group_id1=ca[i][7]

        for j in range(i+1,len(ca)):
          ca[j]=ca[j]
          atm_name2=ca[j][0]
          res_id2=ca[j][1]
          res_name2=ca[j][2]
          chain_id2=ca[j][3]
          x2=float(ca[j][4])
          y2=float(ca[j][5])
          z2=float(ca[j][6])
          group_id2=ca[j][7]
          dist=((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
          #f = open("/content/drive/MyDrive/Antibio/Bin_dist_res/CA_CB_CT_bin_output.txt", "a")
          res = [atm_name1,res_id1,res_name1,chain_id1,group_id1, "   -   " ,atm_name2,res_id2,res_name2,chain_id2,group_id2,"   ----   ",dist ]
          final_op.append(res)
    writecsv(final_op,write_filename)

#Iterating over multiple files and invoking method
for each_file in files:
  path=each_file+".txt"
  file_name=each_file.split('/')[-1]
  target_file="/content/drive/MyDrive/Antibio/Negative_distance_binned2/"+each_file.split('/')[-1]+".txt"
  id=file_name.split('_')[0]
  print(path)
  binning(id,path,target_file)


# Binning the vectors for desired output
files=[]
#with open("/content/drive/MyDrive/Antibio/Final_antibio_pdb_id.txt",'r') as file:
with open("/content/drive/MyDrive/Antibio/Negative_pdb_id.txt",'r') as file:
  for line in file.readlines():
    row=line.split()
    #print(row)
    if len(row)>=0:
      each_file=row[0]
      #print(each_file)
      

     # filename="/content/drive/MyDrive/Antibio/Positive_distance_binned2_final/" +each_file
      filename="/content/drive/MyDrive/Antibio/Negative_distance_binned2/" +each_file
      files.append(filename) 
      #print(files)
# to write it to a new csv file
import csv
def writeBin(rows,file,Bin_details):
  field = ["Residues"]
  for x in Bin_details.keys():
    field.append(x)
  field.append("Distance")
  with open(file, 'w') as f:
        
      # using csv.writer method from CSV package
      write = csv.writer(f)
        
      write.writerow(field)
      write.writerows(rows)


def vector(id,path,write_filename):
  write_data=[]
  res = []
  file_1 = open(path,'r')
  file_1 = file_1.readlines()
  
 

  for i in file_1 :
    Bin_details = {"00CACA":0,"00CACB":0,"00CACT":0,"00CBCB":0,"00CBCT":0,"00CTCT":0,
                  "01CACA":0, "01CACB":0,"01CACT":0,"01CBCB":0,"01CBCT":0,"01CTCT":0,
                  "02CACA":0,"02CACB":0,"02CACT":0,"02CBCB":0,"02CBCT":0,"02CTCT":0,
                  "03CACA":0,"03CACB":0,"03CACT":0,"03CBCB":0,"03CBCT":0,"03CTCT":0,
                  "04CACA":0,"04CACB":0,"04CACT":0,"04CBCB":0,"04CBCT":0,"04CTCT":0,
                  "11CACA":0,"11CACB":0,"11CACT":0,"11CBCB":0,"11CBCT":0,"11CTCT":0,
                  "12CACA":0,"12CACB":0,"12CACT":0,"12CBCB":0,"12CBCT":0,"12CTCT":0,
                  "13CACA":0,"13CACB":0,"13CACT":0,"13CBCB":0,"13CBCT":0,"13CTCT":0,
                  "14CACA":0, "14CACB":0, "14CACT":0,"14CBCB":0,"14CBCT":0,"14CTCT":0,
                  "22CACA":0,"22CACB":0,"22CACT":0, "22CBCB":0, "22CBCT":0,"22CTCT":0,
                  "23CACA":0,"23CACB":0,"23CACT":0, "23CBCB":0, "23CBCT":0,"23CTCT":0,
                  "24CACA":0, "24CACB":0, "24CACT":0,"24CBCB":0,"24CBCT":0,"24CTCT":0, 
                  "33CACA":0,"33CACB":0, "33CACT":0, "33CBCB":0,"33CBCT":0,"33CTCT":0,
                  "34CACA":0,"34CACB":0, "34CACT":0, "34CBCB":0, "34CBCT":0,"34CTCT":0,
                  "44CACA":0, "44CACB":0, "44CACT":0, "44CBCB":0,"44CBCT":0, "44CTCT":0, }
    final_ans=[]
    df=i.split()
    col1=df[0]+' '+df[1]+' '+df[2]+' '+df[6]+' '+df[7]+' '+df[8]+' '
    #col1 = i[0]+' '+i[1]+' '+i[2]+' '+i[6]+' '+i[7]+' '+i[8]+' '
    col1_ind=df[0]
    col2_ind=df[6]
    c1=int(df[4])
    #c1 = i[4]
    #c2 = i[10]
    c2=int(df[10])
    dist=df[-1]
    start_col=str(min(c1,c2))
    end_col=str(max(c1,c2))
    target_col=start_col+end_col+col1_ind+col2_ind
    #print(col1,target_col,dist)
    final_ans.append(col1)
    Bin_details[target_col]= dist
    for j in Bin_details.values():
      final_ans.append(j)
    final_ans.append(dist)
    res.append(final_ans)
  writeBin(res,write_filename,Bin_details)
  #print(res1)

#Iterating over multiple files and invoking method
for each_file in files:
  path=each_file+".txt"
  file_name=each_file.split('/')[-1]
  target_file="/content/drive/MyDrive/Antibio/Negativ_distpop_vector/"+each_file.split('/')[-1]+".csv"
  #target_file="/content/drive/MyDrive/Antibio/Positv_distpop_vector/"+each_file.split('/')[-1]+".csv"
  id=file_name.split('_')[0]
  print(path)
  vector(id,path,target_file)



# Final Mean calculation for all the antibiotic & non-antibiotic class specific to their pockets

files=[]
with open("/content/drive/MyDrive/Antibio/mean_file_id.txt",'r') as file:
  for line in file.readlines():
    row=line.split()
    #print(row)
    if len(row)>=0:
      each_file=row[0]
      #print(each_file)
      

      filename="/content/drive/MyDrive/Antibio/mean_cal_final/" +each_file
      files.append(filename) 
      #print(files)

import csv
def writeBin(rows,file):
  field = ["Pockets"]
  Bin_details = ["00CACA","00CACB","00CACT","00CBCB","00CBCT","00CTCT",
                "01CACA", "01CACB","01CACT","01CBCB","01CBCT","01CTCT",
                 "02CACA","02CACB","02CACT","02CBCB","02CBCT","02CTCT",
                 "03CACA","03CACB","03CACT","03CBCB","03CBCT","03CTCT",
                 "04CACA","04CACB","04CACT","04CBCB","04CBCT","04CTCT",
                 "11CACA","11CACB","11CACT","11CBCB","11CBCT","11CTCT",
                 "12CACA","12CACB","12CACT","12CBCB","12CBCT","12CTCT",
                 "13CACA","13CACB","13CACT","13CBCB","13CBCT","13CTCT",
                 "14CACA", "14CACB", "14CACT","14CBCB","14CBCT","14CTCT",
                 "22CACA","22CACB","22CACT", "22CBCB", "22CBCT","22CTCT",
                 "23CACA","23CACB","23CACT", "23CBCB", "23CBCT","23CTCT",
                 "24CACA", "24CACB", "24CACT","24CBCB","24CBCT","24CTCT", 
                 "33CACA","33CACB", "33CACT", "33CBCB","33CBCT","33CTCT",
                 "34CACA","34CACB", "34CACT", "34CBCB", "34CBCT","34CTCT",
                 "44CACA", "44CACB", "44CACT", "44CBCB","44CBCT", "44CTCT" ]
  for x in Bin_details:
    field.append(x)
  #field.append("Distance")
  with open(file, 'w') as f:
        
      # using csv.writer method from CSV package
      write = csv.writer(f)
        
      write.writerow(field)
      write.writerows(rows)

def mean_cal ():
    #file_1 = open("/content/1d8t_D_B_vector_dis.csv",'r')
    #file_1 = file_1.readlines()
    #df= pd.read_csv("2jq7_C_A_vector_dis.csv")
    #df = df.readlines()
    #df = pd.read_csv(file_1)
    a =file_1.split('/')[-1]
    result =['a']
    df1 = df.replace(0, np.NaN)
    df_final = df1.mean().round(3)
    for i in df_final:
      if math.isnan(i):
         result.append(0)
      else:
          result.append(i)
    writeBin([result],"/content/mean_final.csv")

mean_cal()
