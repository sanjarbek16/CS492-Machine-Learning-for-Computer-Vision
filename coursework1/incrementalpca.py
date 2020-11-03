'''
This is the code for building a face recognition model using Incremental PCA method to dynamically update the model.
Data I am using has 520 face images belonging to 52 different people. I am dividing them into 5 groups, 4 of them for training and 1 for testing.
Each group has 2 pictures belonging to the same person, 104 pictures in total.
First I create a PCA model using one group, then incrementally update the model to add the rest of the training images.
Face recognition accuracy of PCA model trained by only one group is equal to approximately 33.6%. When we add the rest of the training images, accuracy goes up to 52.88%.
Incremental PCA yields lower accuracy compared to batch PCA where the model is trained at once using all the data availabe.

'''
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
from mpl_toolkits.mplot3d import Axes3D
import linecache

mat = scipy.io.loadmat('face.mat')
X = mat['X']
L = mat['l']

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
img = ax1.imshow(np.reshape(X[:,1],(46,56)).T)

training_data1 = np.empty((2576,104))
training_data2 = np.empty((2576,104))
training_data3 = np.empty((2576,104))
training_data4 = np.empty((2576,104))
test_data = np.empty((2576, 104))
training_label1 = []
training_label2 = []
training_label3 = []
training_label4 = []
test_label = []


i=0
j=0
while i!=520:
  training_data1[:, j] = X[:, i]
  training_label1.append(L[:,i][0])
  i+=1
  training_data2[:, j] = X[:, i]
  training_label2.append(L[:,i][0])
  i+=1
  training_data3[:, j] = X[:, i]
  training_label3.append(L[:,i][0])
  i+=1
  training_data4[:, j] = X[:, i]
  training_label4.append(L[:,i][0])
  i+=1
  test_data[:, j] = X[:, i]
  test_label.append(L[:,i][0])
  i+=1
  j+=1
training_data = [training_data2, training_data3, training_data4]
training_labels = [training_label2, training_label3, training_label4]


training_label = training_label1



# calculation of average faces of 4 training groups
avgFace1 = np.mean(training_data1,axis=1).reshape(-1,1)
avgFace2 = np.mean(training_data2,axis=1).reshape(-1,1)
avgFace3 = np.mean(training_data3,axis=1).reshape(-1,1)
avgFace4 = np.mean(training_data4,axis=1).reshape(-1,1)

# list containing last 3 average faces
avg_faces = [avgFace2, avgFace3, avgFace4]

# compute eigenfaces on mean-subtracted training data
A1 = training_data1 - np.tile(avgFace1,training_data1.shape[1])


S1 = np.matmul(A1.T, A1)

S1 /=104

M = 100
N1 = 104
N2 = 104



start = timeit.default_timer()

'''
We start by creating face recognition model using only the first group
'''

##############################################################################################################################
batch1_w, batch1_v = np.linalg.eig(S1)
temp1 = {}

for i in range(len(batch1_w)):
  temp1[batch1_w[i]] = i


# sorting to get top M eigenvalues
sorted_batch1_w = -np.sort(-batch1_w)

topEigenvalues_batch1 = sorted_batch1_w[:M]


lambda1 = np.diag(topEigenvalues_batch1)


# top eigenvectors that correspond to the top eigenvalues
topEigenvectors_batch1 = np.empty([104, M])

for i in range(M):
  topEigenvectors_batch1[:, i] = batch1_v[:, temp1[topEigenvalues_batch1[i]]]
 



# changing eigenvectors from lower dimension to higher dimension
P1 = np.empty([2576, M])

for i in range(M):
  P1[:, i] = A1@topEigenvectors_batch1[:, i]
  P1[:, i] = P1[:, i]/np.linalg.norm(P1[:, i])

##############################################################################################################################


# incrementally adding the rest of the data and updating the model using for loop
for sub in range(3):
  A2 = training_data[sub] - np.tile(avg_faces[sub],training_data[sub].shape[1])
  S2 = np.matmul(A2.T, A2)
  S2 /=104
  batch2_w, batch2_v = np.linalg.eig(S2)
  temp2 = {}
  for i in range(len(batch1_w)):
    temp2[batch2_w[i]] = i

  # sorting to get top M eigenvalues
  sorted_batch2_w = -np.sort(-batch2_w)

  topEigenvalues_batch2 = sorted_batch2_w[:M]

  lambda2 = np.diag(topEigenvalues_batch2)

  # top eigenvectors that correspond to the top eigenvalues
  topEigenvectors_batch2 = np.empty([104, M])
  for i in range(M):
    topEigenvectors_batch2[:, i] = batch2_v[:, temp2[topEigenvalues_batch2[i]]]  



  # changing eigenvectors from lower dimension to higher dimension
  P2 = np.empty([2576, M])
  for i in range(M):
    P2[:, i] = A2@topEigenvectors_batch2[:, i]
    P2[:, i] = P2[:, i]/np.linalg.norm(P2[:, i])

  h = np.concatenate((P1, P2), axis=1)
  dif = avgFace1-avgFace2
  h = np.concatenate((h, dif), axis=1)

  phi, r = np.linalg.qr(h)

  S1 = P1@lambda1@P1.T
  S2 = P2@lambda2@P2.T

  N3 = N1+N2
  avgFace1 = (N1*avgFace1+N2*avg_faces[sub])/N3
  S1 = (N1/N3)*S1+(N2/N3)*S2 + (N1*N2/(N3*N3))*(avgFace1-avgFace2)@((avgFace1-avgFace2).T)

  new = phi.T@S1@phi

  lambda3_arr, R = np.linalg.eig(new)

  P1 = phi@R

  temp = {}
  for i in range(len(lambda3_arr)):
    temp[lambda3_arr[i]] = i

  #sorting to get top eigenvalues
  sorted_batch_w = -np.sort(-lambda3_arr)

  # topEigenvalues2 = sorted_w2[:M]
  topEigenvalues_batch = sorted_batch_w[:200]

  lambda1 = np.diag(topEigenvalues_batch)

  #top eigenvectors that correspond to the top eigenvalues
  topEigenvectors_batch = np.empty([2576, 200])
  for i in range(200):
    topEigenvectors_batch[:, i] = P1[:, temp[topEigenvalues_batch[i]]]

  P1 = topEigenvectors_batch  

  A = np.concatenate((A1, A2), axis=1)
  training_label += training_labels[sub]
  A1 = A
  N1 = N3



# visualization of eigenfaces
fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
img = ax1.imshow(np.reshape(P1[:, 0],(46,56)).T)
img.set_cmap("gray")



stop = timeit.default_timer()
print('Time: ', stop - start)

#reconstruction of the pictures from eigenfaces
w = np.empty((416, 200))
for n in range(416):
  for i in range(200):
    w[n, i] = (A[:, n].T)@P1[:, i]

rec = (w[0, 0]*P1[:, 0]).reshape(-1,1)
for i in range(1, M):
  rec += (P1[:, i]*w[100, i]).reshape(-1,1)


x = avgFace1 + rec.reshape(-1,1)
actual_picture = avgFace1 + A[:,100].reshape(-1,1)



#visualization of reconstructed picture
fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
img = ax1.imshow(np.reshape(x,(46,56)).T)

plt.subplot(122)
plt.imshow(np.reshape(actual_picture,(46,56)).T)

correct = 0





# face recognition error for different values M
error = []
for j in range(200):
    rec1 = (w[300, 0]*P1[:, 0]).reshape(-1,1)
    for i in range(1, j):
        rec1 += (P1[:, i]*w[300, i]).reshape(-1,1)
    difference = rec1.reshape(-1,1) - A[:, 300].reshape(-1,1)
    error.append(np.linalg.norm(difference)/np.linalg.norm(A[:, 300]))


fig = plt.figure()
eig = fig.add_subplot(111)
eig.set_xlabel('M', fontsize = 15)
eig.set_ylabel('Error', fontsize = 15)



eig_x = [x for x in range(1, 201)]
eig_y = error
eig.plot(eig_x, eig_y)

# testing the final model with 104 test images
for test in range(104):
    original_test = test_data[:, test]

    test_face = original_test.reshape(-1,1)-avgFace1

    w_test = np.empty(200)

    for i in range(200):
        w_test[i] = (test_face.T)@P1[:, i]

    temp=22222
    picture = 0
    for n in range(208):
        dif = w_test - w[n, :]
        dif_len = np.linalg.norm(dif)
        if dif_len<temp:
            temp = dif_len
            picture = n

    if training_label[picture]==test_label[test]:
        correct+=1
    



  
accuracy = correct/104

print("Face recognition accuracy of incremental PCA: ", accuracy)


plt.show()