#install -  !pip install biopython 

from Bio import SeqIO

fasta_sequences = SeqIO.parse(open("fastaFilePath"),'fasta')
with open("outputTxtFileName.txt", "a") as myfile:
  i=0
  for fasta in fasta_sequences:
      name, sequence = fasta.id, str(fasta.seq)
      if(name.find("-") is 0): myfile.write(">"+ str(i) + '|0|training\n'  + sequence + '\n')
      elif(name.find("+") is 0): myfile.write(">"+ str(i) + '|1|training\n'  + sequence + '\n')    
      i+=1