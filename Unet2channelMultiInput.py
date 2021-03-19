import numpy as np
import netCDF4 as nc
import scipy.io as sio

pathToTrainImper='/oasis/scratch/comet/ebi/temp_project/SameInitial/dry/betaSigma/set'
pathToTrainPer='/oasis/scratch/comet/ebi/temp_project/SameInitial/dry/CTL/set'

pathToTestImper='/oasis/scratch/comet/ebi/temp_project/SameInitial/dry/betaSigma'
pathToTestPer='/oasis/scratch/comet/ebi/temp_project/SameInitial/dry/CTL/set'


fileList_trainImper = []
fileList_testImper=[]
fileList_trainPVImper=[]
fileList_testPVImper=[]

fileList_trainPer = []
fileList_testPer=[]
fileList_trainPVPer=[]
fileList_testPVPer=[]


for i in range (1,15):
    fileList_trainImper.append (pathToTrainImper + str(i) + '/PSI_output.nc')
    fileList_trainPer.append (pathToTrainPer + str(i) + '/PSI_output.nc')

for i in range (15,21):
    fileList_testImper.append(pathToTestImper + str(i) + '/PSI_output.nc')
    fileList_testPer.append(pathToTestPer + str(i) + '/PSI_output.nc')




import tensorflow as tf
import keras.backend as K
#from data_manager import ClutteredMNIST
#from visualizer import plot_mnist_sample
#from visualizer import print_evaluation
#from visualizer import plot_mnist_grid
import netCDF4
import numpy as np
from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
import keras
from keras.callbacks import History
history = History()

import keras
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, Concatenate, ZeroPadding2D, merge
from keras.models import load_model
from keras.layers import Input
from keras.models import Model
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense

#from utils import get_initial_weights
#from layers import BilinearInterpolation

__version__ = 0.1

lab=0.5

#def stn(input_shape1=(192, 96,2)):
input_shape1=(192, 96,2)
inputs11 = Input(shape=input_shape1)
inputs22 = Input(shape=input_shape1)

conv1 = Convolution2D(32, (5, 5), activation='relu', padding='same')(inputs11)
conv1 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool1)
conv2 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool2)
conv3 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv3)

conv5 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool2)
x = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv5)

up6 = keras.layers.Concatenate(axis=-1)([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x)), conv2])
conv6 = Convolution2D(32, (5, 5), activation='relu', padding='same')(up6)
conv6 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv6)

up7 = keras.layers.Concatenate(axis=-1)([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv1])
conv7 = Convolution2D(32, (5, 5), activation='relu', padding='same')(up7)
conv7 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv7)

### Use tanh in this last layer
conv10 = Convolution2D(2, (5, 5), activation='linear',padding='same')(conv7)

conv1Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(inputs22)
conv1Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv1Per)
pool1Per = MaxPooling2D(pool_size=(2, 2))(conv1Per)

conv2Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool1)
conv2Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv2Per)
pool2Per = MaxPooling2D(pool_size=(2, 2))(conv2Per)

conv3Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool2)
conv3Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv3Per)

conv5Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool2)
xPer = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv5Per)

up6Per = keras.layers.Concatenate(axis=-1)([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(xPer)), conv2Per])
conv6Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(up6Per)
conv6Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv6Per)

up7Per = keras.layers.Concatenate(axis=-1)([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6Per)), conv1Per])
conv7Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(up7Per)
conv7Per = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv7Per)

### Use tanh in this last layer
conv10Per = Convolution2D(2, (5, 5), activation='linear',padding='same')(conv7Per)


model = Model( inputs=[inputs11,inputs22], outputs=[conv10,conv10Per], )

#    return model

#model = stn()
model.summary()
def lossNew(y_pred,y_true):
    o1,o2=y_pred
    i1,i2=y_true
    lab=0.5
    loss = K.mean(lab*tf.keras.losses.MeanSquaredError(o1, i1) + (1-lab)*tf.keras.losses.mean_absolute_error(o2,i2),axis=-1)

#loss = K.mean(lab*tf.keras.losses.MeanSquaredError(conv10, inputs22))# + (1-lab)*tf.keras.losses.mean_absolute_error(inputs44,conv10Per))
#model.add_loss(loss)
model.compile(loss=lossNew,optimizer='adam')
model.summary()


batch_size = 10
num_epochs = 8
lead = 1
count=0
for loop in fileList_trainImper:
    print('******************** counter*************',count)
    print('******************** counter*************',count)
    print('******************** counter*************',count)
    print('******************** counter*************',count)
    print(loop)
    print('******************** counter*************',count)
    print('******************** counter*************',count)
    print('******************** counter*************',count)
    print('******************** counter*************',count)
    spinup=2000
    interval=4000
    cc=0
    while (cc < 2): 
      loopPer=fileList_trainPer[count]
      File=nc.Dataset(loop)
      FilePer=nc.Dataset(loopPer)

      if spinup<6000:
         Z=File['PSI'][spinup:spinup+interval,:,:,:]
         ZPer=FilePer['PSI'][spinup:spinup+interval,:,:,:]
         if spinup==2000:
           cc=0
         else:
           cc=1
      else:
        Z=File['PSI'][spinup:,:,:,:]
        ZPer=FilePer['PSI'][spinup:spinup+intervalPer,:,:,:]
        cc=2

      print(cc)
      print(spinup)
      trainN=int(np.size(Z,0)-np.round(np.size(Z,0)/6))
      Zup=Z[:,0,:,:]
      Zdown=Z[:,1,:,:]
      ZupPer=ZPer[:,0,:,:]
      ZdownPer=ZPer[:,1,:,:]

      del Z,File
      tDim=np.size(Zup,axis=0)
      ny=np.size(Zup,axis=1)
      nx=np.size(Zup,axis=2)
        
      Mup=np.mean(Zup.flatten())
      Mdown=np.mean(Zdown.flatten())
      sdevup=np.std(Zup.flatten())
      sdevdown=np.std(Zdown.flatten())
      Zup=(Zup-Mup)/sdevup
      Zdown=(Zdown-Mdown)/sdevdown
      ZupPer=(ZupPer-Mup)/sdevup
      ZdownPer=(ZdownPer-Mdown)/sdevdown
      
      Z = np.zeros( ( ( tDim , ny , nx,2 ) ) )
      Perdata = np.zeros( ( ( tDim , ny , nx,2 ) ) )

      for dd in range (0, np.size(ZupPer,axis=0),100):
          Perdata[dd,:,:,0] =ZupPer[dd,:,:]
          Perdata[dd,:,:,1] =ZdownPer[dd,:,:]
          Perdata[dd-1,:,:,0] =Zup[dd-1,:,:]
          Perdata[dd-1,:,:,1] =Zdown[dd-1,:,:]
 
      

      Z[:,:,:,0]=Zup
      Z[:,:,:,1]=Zdown 
       
      del Zup,Zdown
      ImPerx_train=Z[0:trainN,:,:,:]
      Perx_train=Perdata[0:trainN,:,:,:]
      ImPery_train=Z[lead:trainN+lead,:,:,:]
      Pery_train=Perdata[lead:trainN+lead,:,:,:]

      x_val= Z[trainN+lead:np.size(Z,0)-lead,:,:,:]
      Perx_val= Pery_train[trainN+lead:np.size(Z,0)-lead,:,:,:]
      y_val= Z[trainN+lead*2:np.size(Z,0),:,:,:]
      Pery_val=Pery_train[trainN+lead*2:np.size(Z,0),:,:,:]


      if (count>0 or cc>0):

          model = stn()
          model.compile(loss='mse', optimizer='adam')
          model.load_weights('Multibest_weights_lead1.h5')
          hist = model.fit([ImPerx_train, Perx_train],[ImPery_train,Pery_train],
                        batch_size = batch_size,
                verbose=1,
                epochs = 20,
                validation_data=([x_val,Perx_val],[y_val,Pery_val]),shuffle=True,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=5, # just to make sure we use a lot of patience before stopping
                                            verbose=0, mode='auto'),
                          keras.callbacks.ModelCheckpoint('Multibest_weights_lead1.h5', monitor='val_loss',
                                                        verbose=1, save_best_only=True,
                                                        save_weights_only=True, mode='auto', period=1),history]
                )
      else:
          hist = model.fit([ImPerx_train, Perx_train],[ImPery_train,Pery_train],
                        batch_size = batch_size,
              verbose=1,
              epochs = 20,
              validation_data=([x_val,Perx_val],[y_val,Pery_val]),shuffle=True,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=0,
                                           patience=5, # just to make sure we use a lot of patience before stopping
                                           verbose=0, mode='auto'),
                        keras.callbacks.ModelCheckpoint('Multibest_weights_lead1.h5', monitor='val_loss',
                                                      verbose=1, save_best_only=True,
                                                      save_weights_only=True, mode='auto', period=1),history]
              )
            

      spinup=spinup+interval
    count=count+1
	
model = stn()
loss = K.mean(lab*mse(conv10, inputs3) + (1-lab)*mae(inputs2,conv10Per))
model.add_loss(loss)
model.compile(optimizer='adam')
model.summary()

lead=1
model.load_weights('Multibest_weights_lead' + str(lead) + '.h5')
count=0

for loop in fileList_testPer:

    F=nc.Dataset(loop)
    cc=0
    spinup=2000
    interval=500
    while (cc < 2): 
            if spinup<9500:
              Z=F['PSI'][spinup:spinup+interval,:,:,:]
              if spinup==2000:
                cc=0
              else:
                cc=1
            else:
              Z=F['PSI'][spinup:,:,:,:]
              cc=2
            
            print(cc)
            print(spinup)            
            del Z,File
              
            Zup=Z[:,0,:,:]
            Zdown=Z[:,1,:,:]
            del Z
            tDim=np.size(Zup,axis=0)
            ny=np.size(Zup,axis=1)
            nx=np.size(Zup,axis=2)
        
            Mup=np.mean(Zup.flatten())
            Mdown=np.mean(Zdown.flatten())
            sdevup=np.std(Zup.flatten())
            sdevdown=np.std(Zdown.flatten())
            Zup=(Zup-Mup)/sdevup
            Zdown=(Zdown-Mdown)/sdevdown

            Z = np.zeros( ( ( tDim , ny , nx,2 ) ) )
            Z[:,:,:,0]=Zup
            Z[:,:,:,1]=Zdown
            del Zup,Zdown

            testN=np.size(Z,0)
            x_test=Z[0:testN,:,:,:]
            y_test=Z[lead:testN+lead,:,:,:]
            pred=np.zeros([testN,ny,nx,2])

            for k in range (0, testN):
                if(k==0):
                  o1,o2=model.predict(x_test[k,:,:,:].reshape([1,ny,nx,2]),x_test[k,:,:,:].reshape([1,ny,nx,2])).reshape([ny,nx,2])
                  pred[k,:,:,:]=o1
                else:
                  o1,o2=model.predict(pred[k-1,:,:,:].reshape([1,ny,nx,2]),pred[k-1,:,:,:].reshape([1,ny,nx,2])).reshape([ny,nx,2])
                  pred[k,:,:,:]=o1

            count+=1
            pred[:,:,:,0]=(pred[:,:,:,0]*sdevup)+Mup
            pred[:,:,:,1]=(pred[:,:,:,1]*sdevdown)+Mdown

            y_test[:,:,:,0]=(y_test[:,:,:,0]*sdevup)+Mup
            y_test[:,:,:,1]=(y_test[:,:,:,1]*sdevdown)+Mdown

            sio.savemat('ERA5_1_hr_UNet_noSTN'+ str(count) + '.mat',dict([('prediction',pred),('truth',y_test)]))
            spinup=spinup+interval
print('Finished writing File')


del pred,y_test


def mean2(x):
    y = np.sum(x) / np.size(x);
    return y
def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r

F=nc.Dataset('/oasis/scratch/comet/ebi/temp_project/SameInitial/dry/CTL/set15/PSI_output.nc')

psi=np.asarray(F['PSI'])
psi=psi[2500:,1,:,:]
MEAN_L1 = np.mean(psi,axis=0)
AnomalyAll=np.zeros([100,192,96])

for k in range(0,99):
    AnomalyAll[k,:,:]=psi[k,:,:]-MEAN_L1

sio.savemat('DownmeanAno.mat',dict([('Psimean',MEAN_L1),('Anomaly',AnomalyAll)]))


for k in range(1,94):
   file=sio.loadmat('ERA5_1_hr_UNet_noSTN'+ str(k) + '.mat')
   prediction=file['prediction']
   truth=file['truth']
   rmse=np.zeros([200])
   ACC=np.zeros([200])
   for d in range(0,199):
       prediction1=prediction[d,:,:,1].flatten()-MEAN_L1.flatten()
       truth1=truth[d,:,:,1].flatten()-MEAN_L1.flatten()
       rmse[d]=np.sqrt(np.mean((prediction[d,:,:,1].flatten()-truth[d,:,:,1].flatten())**2))
       ACC[d]=corr2(prediction1,truth1)
   sio.savemat('DownrmseACC'+ str(k) + '.mat',dict([('rmse',rmse),('ACC',ACC)]))

del prediction1,prediction,truth1,truth,MEAN_L1,psi

F=nc.Dataset('/oasis/scratch/comet/ebi/temp_project/SameInitial/dry/CTL/set15/PSI_output.nc')

psi=np.asarray(F['PSI'])
psi=psi[2500:,0,:,:]
MEAN_L1 = np.mean(psi,axis=0)
AnomalyAll=np.zeros([100,192,96])

for k in range(0,99):
    AnomalyAll[k,:,:]=psi[k,:,:]-MEAN_L1

sio.savemat('UpmeanAno.mat',dict([('Psimean',MEAN_L1),('Anomaly',AnomalyAll)]))


for k in range(1,94):
   file=sio.loadmat('ERA5_1_hr_UNet_noSTN'+ str(k) + '.mat')
   prediction=file['prediction']
   truth=file['truth']
   rmse=np.zeros([200])
   ACC=np.zeros([200])
   for d in range(0,199):
       prediction1=prediction[d,:,:,0].flatten()-MEAN_L1.flatten()
       truth1=truth[d,:,:,0].flatten()-MEAN_L1.flatten()
       rmse[d]=np.sqrt(np.mean((prediction[d,:,:,0].flatten()-truth[d,:,:,0].flatten())**2))
       ACC[d]=corr2(prediction1,truth1)
   sio.savemat('UprmseACC'+ str(k) + '.mat',dict([('rmse',rmse),('ACC',ACC)]))

