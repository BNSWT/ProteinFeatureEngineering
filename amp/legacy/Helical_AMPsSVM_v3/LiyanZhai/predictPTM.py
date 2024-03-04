import sys 
sys.path.append("/home/e2-305/Data/Helixml/Helical_AMPsSVM_v3/Helical_AMPsSVM_v3/LiyanZhai")
import pandas as pd
import numpy as np
import argparse
import csv
from theano import config
config.warn.round=False
import warnings
warnings.filterwarnings('ignore')


def predPTM(sequence,residue="S,Y,T",score=0.6):
    ''' 
    predict phosphorylation
    Returns phosphorylation sites with scores exceeding a certain value
    if empty filter_result.empty return true
    '''
    predicttype="general"; 
    filter_result=pd.DataFrame()
    if predicttype == 'general': #prediction for general phosphorylation
        from methods.DProcess import convertRawToXY
        from methods.EXtractfragment_sort import extractFragforPredict
        from methods.multiCNN import MultiCNN
        nclass=5
        window=16
        results_ST=None
        results_Y=None
        #################for S and T
        residues=residue.split(",")
        if("Y" in residues):
          residues.remove("Y")
        
        if("S" in residues or "T" in residues) and ("S" in sequence or "T" in sequence):
            #    print "General phosphorylation prediction for S or T: \n"        
               testfrag,ids,poses,focuses=extractFragforPredict(sequence,window,'-',focus=residues)
               testX,testY = convertRawToXY(testfrag.as_matrix(),codingMode=0) 
               predictproba=np.zeros((testX.shape[0],2))
               models=MultiCNN(testX,testY,nb_epoch=1,predict=True)# only to get config
               model="./models/models_ST_HDF5model_class"
               for bt in range(nclass):
                   models.load_weights(model+str(bt))
                   predictproba+=models.predict(testX)
                #    print("Done predicting by model of class "+str(bt)+"\n");
               
               predictproba=predictproba/nclass;
               poses=poses+1;
               results_ST=np.column_stack((ids,poses,focuses,predictproba[:,1]))
               result=pd.DataFrame(results_ST)

               filter_result=result[result.iloc[:,3]>score]
               filter_result[0]=sequence
 

            #    result.to_csv("general_phosphorylation_SorT.txt", index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
              
        #########for Y################
        residues=residue.split(",")
        if("Y" in residues) and ("Y" in sequence):
        #    print "General phosphorylation prediction for Y: \n"        
           testfrag,ids,poses,focuses=extractFragforPredict(sequence,window,'-',focus=("Y"))
           testX,testY = convertRawToXY(testfrag.as_matrix(),codingMode=0) 
           predictproba=np.zeros((testX.shape[0],2))
           models=MultiCNN(testX,testY,nb_epoch=1,predict=True)# only to get config
           model="./models/models_Y_HDF5model_"
           nclass_init=5;
           nclass=3;
           for ini in range(nclass_init):
                   for bt in range(nclass):
                           models.load_weights(model+'ini'+str(ini)+'_class'+str(bt))
                           predictproba+=models.predict(testX)
                        #    print("Done predicting by model of class "+str(bt)+" and initial class "+str(ini)+" \n");
           
           predictproba=predictproba/(nclass*nclass_init);
           poses=poses+1;
           results_Y=np.column_stack((ids,poses,focuses,predictproba[:,1]))
           result=pd.DataFrame(results_Y)
           result[0]=sequence
           filter_result=filter_result.append(result[result.iloc[:,3]>score])
        #    result.to_csv("general_phosphorylation_Y.txt", index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
        
        # print "Successfully predicted for general phosphorylation !\n";


    return filter_result
     


# predPTM("MEVQLGLGRVYPRPPSKTYRGAFQNLFQSVREVIQNPGPRHPEAASAAPPGASLLLLQQQ")
# predPTM("GAYGGYGGGVGGWFGGMGGGGAYGGYGGGVGGWFGGMGGG")