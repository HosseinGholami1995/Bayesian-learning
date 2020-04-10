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


#main program -----k fold 
from scipy.stats import multivariate_normal
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
    
   #learning part    
    st_data=df.iloc[spliter(0,50,train_index)]
    st_mean=st_data.mean()
    st_cov=st_data.cov().values  
   
    
    
    vs_data=df.iloc[spliter(50,100,train_index)]
    vs_mean=vs_data.mean()
    vs_cov=vs_data.cov().values
    
    
    vg_data=df.iloc[spliter(100,150,train_index)]
    vg_mean=vg_data.mean()
    vg_cov=vg_data.cov().values  
    
    #now we test data :
 
    test_data=df.values[test_index]
    test=np.delete(test_data,4,axis=1)
    
    
    st_res=multivariate_normal.logpdf(test,mean=st_mean,cov=st_cov)
    vs_res=multivariate_normal.logpdf(test,mean=vs_mean,cov=vs_cov)
    vg_res=multivariate_normal.logpdf(test,mean=vg_mean,cov=vg_cov)
    
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

    
    
    
    
    
    
    
    
    
    
    