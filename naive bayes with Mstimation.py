import numpy as np
import pandas as pd
path='F:\Data\iris.csv'
df=pd.read_csv(path)

Feature =['sl','sw','pl','pw']
lable =['cl']

def spliter(n1,n2,test_list):
    k=list()
    for i in test_list:
        if(i<n2)&(i>n1):            
            k.append(i)
    return k

def independecy(cov_matrix):
    for i in range(len(cov_matrix)):
        for j in range(len(cov_matrix)):
            if (i!=j):    
                cov_matrix[i][j]= 0
    return cov_matrix

#for soft estimation
m=3

#main program -----k fold 
from sklearn.model_selection import KFold

kf = KFold (n_splits=3,shuffle=True,random_state=np.random)

total_accuracy=list()

for train_index,test_index in kf.split(df):
    
    #print('train index\n',train_index)
    print('train index of st\n',spliter(0,50,train_index))
    print('train index of vs\n',spliter(50,100,train_index))
    print('train index of vg\n',spliter(100,150,train_index))
    print('test index\n',test_index)
    print('_______________________________________________________\n')
    
    st_data=df.iloc[spliter(0,50,train_index)]
    vs_data=df.iloc[spliter(50,100,train_index)]
    vg_data=df.iloc[spliter(100,150,train_index)]
    
    #learning part
    st_dict=list()
    vs_dict=list()
    vg_dict=list()
    
    for feature in Feature:
        st_dict.append((st_data.groupby(feature).size()))
        vs_dict.append((vs_data.groupby(feature).size()))
        vg_dict.append((vg_data.groupby(feature).size()))
        
    #now we test data :
 
    test_data=df.values[test_index]
    test=np.delete(test_data,4,axis=1)
    
    st_res=list()
    vs_res=list()
    vg_res=list()
    
    for i in range(0,len(test)):
        a=1
        #___________________________st func
        for j in range(len(Feature)):
            if test[i][j] in st_dict[j] :
            
                a=a*(st_dict[j][test[i][j]]+(3*m))/(m+len(st_data))
                
            else:
                
                a=a*((3*m)/(m+len(st_data)))
                
        st_res.append(a)
        #_____________________________vs func
        a=1
        for j in range(len(Feature)):
            if test[i][j] in vs_dict[j] :
            
                a=a*(vs_dict[j][test[i][j]]+(3*m))/(m+len(vs_data))
                
            else:
                
                a=a*((3*m)/(m+len(vs_data)))
        vs_res.append(a)
        #_______________________________vg func
        a=1
        for j in range(len(Feature)):
            if test[i][j] in vg_dict[j] :
            
                a=a*(vg_dict[j][test[i][j]]+(3*m))/(m+len(vg_data))
                
            else:
                
                a=a*((3*m)/(m+len(vg_data)))
                
        vg_res.append(a)
        
    result=list()
    
    for i in range(0,len(test)):
        if(st_res[i]>vs_res[i])&(st_res[i]>vg_res[i]):
            if test_data[i][4]=='st':
                result.append('T')
            else: 
                result.append('F')
            
        if(vs_res[i]>st_res[i])&(vs_res[i]>vg_res[i]):
            if test_data[i][4]=='vs':
                result.append('T')
            else: 
                result.append('F')
                        
            
        if(vg_res[i]>st_res[i])&(vg_res[i]>vs_res[i]):
            if test_data[i][4]=='vg':
                result.append('T')
            else: 
                result.append('F')
                        
    print('the number of True is: ',result.count('T'),'\nthe number of False is :',result.count('F'),'\n')
    accuracy=result.count('T')/(result.count('T')+result.count('F'))
    print('the accuracy : ',accuracy,'\n')
    total_accuracy.append(accuracy)    

end_accuracy=sum(total_accuracy)/len(total_accuracy)

print('====================================================================')
print('total accuracy is :',end_accuracy)

