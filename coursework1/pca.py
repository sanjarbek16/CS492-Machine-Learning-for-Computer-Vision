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



training_data = np.empty((2576,416))
test_data = np.empty((2576, 104))
training_label = []
test_label = []
j = 0
k = 0
for i in range(520):
    if i%10<8:
        training_data[:,j] = X[:,i]
        training_label.append(L[:,i][0])
        j+=1
    else:
        test_data[:, k] = X[:,i]
        test_label.append(L[:,i][0])
        k+=1


avgFace = np.mean(training_data,axis=1).reshape(-1,1)
fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
img = ax1.imshow(np.reshape(avgFace,(46,56)).T)

# compute eigenfaces on mean-subtracted training data
A = training_data - np.tile(avgFace,training_data.shape[1])
S1 = np.matmul(A, A.T)
S2 = np.matmul(A.T, A)
S1 /=416
S2 /=416


M = 100
N = 416


'''
first method (higher dimensionality method of finding eigenfaces)
##################################################################################################################
'''
start = timeit.default_timer()


w1, u = np.linalg.eig(S1)
temp = {}
for i in range(len(w1)):
    temp[w1[i]] = i


zeros = 0
#sorting to get top eigenvalues and to find number of zero eigenvalues
sorted_w1 = -np.sort(-w1)
for eigenvalue in sorted_w1:
    if round(eigenvalue, 4)==0.0:
        zeros+=1
print("Number of zero eigenvalues in 1st method: ", zeros)


topEigenvalues1 = sorted_w1[:M]


#top eigenvectors that correspond to the top eigenvalues
topEigenvectors1 = np.empty([2576, M])
for i in range(M):
    topEigenvectors1[:, i] = u[:, temp[topEigenvalues1[i]]]


#visualization of eigenfaces
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 3, 1)
img = ax1.imshow(np.reshape(topEigenvectors1[:, 3],(46,56)).T)



img2 = fig1.add_subplot(1, 3, 2)
img2.imshow(np.reshape(topEigenvectors1[:, 4],(46,56)).T)



img3 = fig1.add_subplot(1, 3, 3)
img3.imshow(np.reshape(topEigenvectors1[:, 5],(46,56)).T)



stop = timeit.default_timer()
print('High dimensional PCA computation time: ', stop - start)



'''
##################################################################################################################
'''



'''
low-dimensional computation method
##################################################################################################################
'''
start = timeit.default_timer()

w2, v = np.linalg.eig(S2)
temp = {}
for i in range(len(w2)):
    temp[w2[i]] = i

#sorting to get top eigenvalues
sorted_w2 = -np.sort(-w2)
topEigenvalues2 = sorted_w2[:M]

zeros = 0
for eigenvalue in sorted_w2:
    if round(eigenvalue, 4)==0.0:
        zeros+=1
print("Number of zero eigenvalues in 2nd method: ", zeros)

#top eigenvectors that correspond to the top eigenvalues
topEigenvectors2 = np.empty([416, M])
for i in range(M):
    topEigenvectors2[:, i] = v[:, temp[topEigenvalues2[i]]]



#changing eigenvectors from lower dimension to higher dimension
higherDimEigvec = np.empty([2576, M])
for i in range(M):
    higherDimEigvec[:, i] = A@topEigenvectors2[:, i]
    higherDimEigvec[:, i] = higherDimEigvec[:, i]/np.linalg.norm(higherDimEigvec[:, i])



#visualization of eigenfaces
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 3, 1)
img = ax1.imshow(np.reshape(higherDimEigvec[:, 3],(46,56)).T)



img2 = fig1.add_subplot(1, 3, 2)
img2.imshow(np.reshape(higherDimEigvec[:, 4],(46,56)).T)



img3 = fig1.add_subplot(1, 3, 3)
img3.imshow(np.reshape(higherDimEigvec[:, 5],(46,56)).T)




stop = timeit.default_timer()
print('Low dimensional PCA computation time: ', stop - start)

'''
##################################################################################################################
'''

accuracy = 0

# Face recognition accuracy check for low dimensional PCA computation
correct = 0
w = np.empty((N, M))
w_higher = np.empty((N, M))
for n in range(N):
    for i in range(M):
        w[n, i] = (A[:, n].T)@higherDimEigvec[:, i]
        w_higher[n, i] = (A[:, n].T)@topEigenvectors1[:, i]

for test in range(104):
    original_test = test_data[:, test]

    test_face = original_test.reshape(-1,1)-avgFace

    w_test = np.empty(M)

    for i in range(M):
        w_test[i] = (test_face.T)@higherDimEigvec[:, i]

    temp=22222
    picture = 0
    for n in range(N):
        dif = w_test - w[n, :]
        dif_len = np.linalg.norm(dif)
        if dif_len<temp:
            temp = dif_len
            picture = n

    if training_label[picture]==test_label[test]:
        correct+=1
    

    
accuracy = correct/104

print("Face recognition accuracy of low dimesional PCA computation: ", accuracy, "%")


correct2 = 0

# Face recognition accuracy check for high dimensional PCA computation
for test in range(104):
    original_test = test_data[:, test]

    test_face = original_test.reshape(-1,1)-avgFace

    w_test = np.empty(M)

    for i in range(M):
        w_test[i] = (test_face.T)@topEigenvectors1[:, i]

    temp=22222
    picture = 0
    for n in range(N):
        dif = w_test - w_higher[n, :]
        dif_len = np.linalg.norm(dif)
        if dif_len<temp:
            temp = dif_len
            picture = n

    if training_label[picture]==test_label[test]:
        correct2+=1

accuracy = correct2/104

print("Face recognition accuracy of low dimensional PCA computation: ", accuracy, "%")



# visualization of projection of images in the first 4 class in 3D eigenspace using top 3 eigenvalues
graph_data = []
for i in range(24, 56):
    graph_data.append(w[i, :3])



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', fc='white')


x=[[], [], [], []]
y=[[], [], [], []]
z=[[], [], [], []]

for i in range(32):
    x[i//8].append(graph_data[i][0])
    y[i//8].append(graph_data[i][1])
    z[i//8].append(graph_data[i][2])


for n, label in [(0, 'o'), (1, '*'), (2, '^'), (3, 'x')]:
    xs = x[n]
    ys = y[n]
    zs = z[n]
    ax.scatter(xs, ys, zs, marker = label, s=40)

ax.set_xlabel('u1', fontsize = 15)
ax.set_ylabel('u2', fontsize = 15)
ax.set_zlabel('u3', fontsize = 15)




plt.show()
