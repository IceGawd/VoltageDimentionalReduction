import kagglehub
from time import time

t1=time()
path = kagglehub.dataset_download("thanakomsn/glove6b300dtxt")
glove_path = path + "/glove.6B.300d.txt" 
with open(glove_path,'r') as file:
    i=0
    for line in file:
        i+=1
        print(i,end='\r')
t2=time()
print()
print(t2-t1)
