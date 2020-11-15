# PCA_LDA technique of image recognition
# Written by Sanjarbek Rakhmonov and Webi Tesfaye for CS492 @ KAIST

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Insignificant Comlplex numbers will be discarded.
# Ignore the warning.
import warnings
warnings.filterwarnings("ignore")

data_path = ''
data_name = 'face.mat'

try:
	mat = scipy.io.loadmat(data_path + 'face.mat')
	print("\nSuccesfully loaded the file" + data_name +"\n")
except:
	print("Error: File not found. Please set the path correctly")
	exit()

# Get the images in each column of matrix X
# Get the labels L
X = mat['X']
L = mat['l']

# Show images
show_images = True # False

# Get the dimension of one image representation
dimension = X.shape[0]
total_num_images = X.shape[1]
image_per_class = 10
num_of_classes = total_num_images//image_per_class

# Set the training and testing image numbers
training_size = 416
test_size = 104

# Initialize the training and test data along with the labels
training_data = np.empty((dimension,training_size))
test_data = np.empty((dimension, test_size))
training_label = []
test_label = []

# Collect the training and test data
j = 0
k = 0
for i in range(total_num_images):
  if i%10<8:
    training_data[:,j] = X[:,i]
    training_label.append(L[:,i])
    j+=1
  else:
    test_data[:, k] = X[:,i]
    test_label.append(L[:,i])
    k+=1
print("Training and Test images are Collected\n")
print(" "*4, training_size, "images for training")
print(" "*4, test_size, "images for testing")
print("-"*40+'\n')


# Calculate the Average per Class
AvgOfClasses = np.empty((dimension,52))
for i in range(52):
  avg = np.sum(X[:,i*10:i*10+8],axis = 1)/8
  AvgOfClasses[:,i] = avg
# mean of the total training data
m = np.mean(AvgOfClasses,axis = 1).reshape(-1,1)
print("Average of classes and total mean is calculated\n")

if show_images:
	print("Showing Average Face")
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	avg_img = ax1.imshow(np.reshape(m,(46,56)).T)
	plt.title("Mean face")
	plt.show()


# with-in class variance
print("Getting the with-in class scatter matrix...\n")
S_w = np.empty((dimension,dimension))
for i in range(52):
  for image in range(i*8,i*8 + 8):
    diff = (training_data[:,image] - AvgOfClasses[:,i]).reshape(-1,1)
    S_w += diff@diff.T
print("Done!\n")

# between class variance
print("Getting the between-class scatter matrix...\n")
S_b = np.zeros((dimension,dimension))
m = np.mean(AvgOfClasses,axis = 1).reshape(-1,1)
for i in range(52):
  diff = (AvgOfClasses[:,i].reshape(-1,1) - m).reshape(-1,1) 
  S_b += 8*(diff@diff.T)
print("Done!\n")

# The ranks of Scatter matrices
print("Calculating the ranks of Scatter matrices...")
rank_SW = np.linalg.matrix_rank(S_w)
print(" "*4, "1.	Rank of the with-in class matrix is" , rank_SW)
rank_SB = np.linalg.matrix_rank(S_b)
print(" "*4, "2.	Rank of the between-class matrix is", rank_SB,"\n")

# Set the M_pca and M_lda
M_pca = 145 # 145 ----- the best 97/7
M_lda = 51

# The actual labels and the predicted labels
actual = []
result = []

# Calculate A to find S2
A = training_data - np.tile(m,training_data.shape[1])
S2 = np.matmul(A.T, A)/ training_size

# calculate the Eigen vectors using lower dimension S2
#first method (higher dimensionality method of finding eigenfaces)
w2, v = np.linalg.eig(S2)
temp = {}
for i in range(len(w2)):
  temp[w2[i]] = i

#sorting to get top eigenvalues
sorted_w2 = -np.sort(-w2)
topEigenvalues2 = sorted_w2[:M_pca]


#top eigenvectors that correspond to the top eigenvalues
topEigenvectors2 = np.empty([training_size, M_pca])
for i in range(M_pca):
  topEigenvectors2[:, i] = v[:, temp[topEigenvalues2[i]]] 


#changing eigenvectors from lower dimension to higher dimension
higherDimEigvec = np.empty([dimension, M_pca])
for i in range(M_pca):
  higherDimEigvec[:, i] = A@topEigenvectors2[:, i]
  higherDimEigvec[:, i] = higherDimEigvec[:, i]/np.linalg.norm(higherDimEigvec[:, i])
temp_SB = higherDimEigvec.T @ S_b @ higherDimEigvec
temp_SW = higherDimEigvec.T @ S_w @ higherDimEigvec

print("PCA is completed\n")
# temporary s_w formed because of the singularity of the first sw
temp_S = np.linalg.inv(temp_SW) @ temp_SB
w_lda, u_lda = np.linalg.eig(temp_S)
temp2 = {}
for i in range(len(w_lda)):
  temp2[w_lda[i]] = i  
sorted_w_lda = -np.sort(-w_lda)
topEigenvalues_lda = sorted_w_lda[:M_lda]

# top eigenvectors that correspond to the top eigenvalues
topEigenvectors_lda = np.empty((M_pca, M_lda))
for i in range(M_lda):
  topEigenvectors_lda[:, i] = u_lda[:,temp2[topEigenvalues_lda[i]]] 

# Combine both Eigen vectors to make the final Eigen vector representation
combined_eig = higherDimEigvec @ topEigenvectors_lda

# show one fisherface
if show_images:
	print("Showing Fisherface")
	oneeig = combined_eig[:,1]
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	img = ax1.imshow(np.reshape(oneeig,(46,56)).T)
	plt.title("fisherface of M_pca {:d} and M_lda {:d}".format(M_pca,M_lda))
	plt.show()
	# img.set_cmap('gray')

print("PCA_LDA is completed\n")
# The projection of each class mean on the Eigenspace
W_of_classes = np.zeros((M_lda,num_of_classes))
for class_index in range(num_of_classes):
  for ind in range(M_lda):
    W_of_classes[ind,class_index] = combined_eig[:,ind] @ AvgOfClasses[:,class_index].reshape(-1,1)


failed = 0
gotIt = 0
for test_index in range (test_size):
  # load the test image
  test_image = test_data[:,test_index].reshape(-1,1)
  # get the label of the test image
  test_label = test_index//2

  # calculate the projection of test image
  W_of_test_image = np.zeros((M_lda,1))
  for ind in range(M_lda):
    W_of_test_image[ind,0] = combined_eig[:,ind]@test_image

  # compare the test image projection with other classes'
  predicted = float("inf")
  predicted_class = -1
  for class_index in range(num_of_classes):
    dif = W_of_test_image.T - W_of_classes[:, class_index].T
    dif_len = np.linalg.norm(dif)
    if predicted > dif_len:
      predicted = dif_len
      predicted_class = class_index
  if predicted_class == test_label:
    gotIt+=1
  else:
    #print("failed for", predicted_class,test_label)
    failed +=1
    actual.append(test_label)
    result.append(predicted_class)
print("M_pca =", M_pca, "and M_lda =", M_lda)
print("got", gotIt,"and failed ", failed,"accuracy = {:.2f}".format((gotIt /test_size)*100), "%\n")



loop_through = False
# Try on different combinations of M_pca and M_lda
if loop_through:
	mpca_index = 0
	M1_list = list(range(140,150))
	M2_list = list(range(45,52))
	table = np.zeros((len(M1_list),len(M2_list)))
	for M1 in M1_list:
	  mlda_index = 0
	  for M2 in M2_list:

	    print("M_pca = ",M1,"M_lda =", M2,end=" ")
	    M_pca = M1 # 145 ----- the best 97/7
	    M_lda = M2
	    
	    
	    actual = []
	    result = []
	    # Calculate A to find S2
	    A = training_data - np.tile(m,training_data.shape[1])
	    S2 = np.matmul(A.T, A)/ training_size
	    # calculate the Eigen vectors using lower dimension S2
	    #first method (higher dimensionality method of finding eigenfaces)
	    w2, v = np.linalg.eig(S2)
	    temp = {}
	    for i in range(len(w2)):
	      temp[w2[i]] = i

	    #sorting to get top eigenvalues
	    sorted_w2 = -np.sort(-w2)
	    topEigenvalues2 = sorted_w2[:M_pca]


	    #top eigenvectors that correspond to the top eigenvalues
	    topEigenvectors2 = np.empty([training_size, M_pca])
	    for i in range(M_pca):
	      topEigenvectors2[:, i] = v[:, temp[topEigenvalues2[i]]] 


	    #changing eigenvectors from lower dimension to higher dimension
	    higherDimEigvec = np.empty([dimension, M_pca])
	    for i in range(M_pca):
	      higherDimEigvec[:, i] = A@topEigenvectors2[:, i]
	      higherDimEigvec[:, i] = higherDimEigvec[:, i]/np.linalg.norm(higherDimEigvec[:, i])
	    temp_SB = higherDimEigvec.T @ S_b @ higherDimEigvec
	    temp_SW = higherDimEigvec.T @ S_w @ higherDimEigvec

	    # temporary s_w formed because of the singularity of the first sw
	    temp_S = np.linalg.inv(temp_SW) @ temp_SB

	    w_lda, u_lda = np.linalg.eig(temp_S)
	    temp2 = {}
	    for i in range(len(w_lda)):
	      temp2[w_lda[i]] = i  
	    sorted_w_lda = -np.sort(-w_lda)
	    topEigenvalues_lda = sorted_w_lda[:M_lda]

	    # top eigenvectors that correspond to the top eigenvalues

	    topEigenvectors_lda = np.empty((M_pca, M_lda))
	    for i in range(M_lda):
	      topEigenvectors_lda[:, i] = u_lda[:,temp2[topEigenvalues_lda[i]]] 

	    # combine both eigenspaces to form the eigenspace for the classification
	    combined_eig = higherDimEigvec @ topEigenvectors_lda

	    if show_images:
	    	oneeig = combined_eig[:,1]
	    	fig1 = plt.figure()
	    	ax1 = fig1.add_subplot(111)
	    	img = ax1.imshow(np.reshape(oneeig,(46,56)).T)
	    	plt.title("fisherface of M_pca {:d} and M_lda {:d}".format(M_pca,M_lda))
	    	plt.show()
	    	# img.set_cmap('gray')
	    
	    
	    W_of_classes = np.zeros((M_lda,num_of_classes))
	    for class_index in range(num_of_classes):
	      for ind in range(M_lda):
	        W_of_classes[ind,class_index] = combined_eig[:,ind] @ AvgOfClasses[:,class_index].reshape(-1,1)

	    failed = 0
	    gotIt = 0
	    for test_index in range (test_size):
	      # load the test image
	      test_image = test_data[:,test_index].reshape(-1,1)
	      # get the label of the test image
	      test_label = test_index//2

	      # calculate the w of test image
	      W_of_test_image = np.zeros((M_lda,1))
	      for ind in range(M_lda):
	        W_of_test_image[ind,0] = combined_eig[:,ind]@test_image

	      # compare the test image 'w' with other classes'
	      predicted = float("inf")
	      predicted_class = -1
	      for class_index in range(num_of_classes):
	        dif = W_of_test_image.T - W_of_classes[:, class_index].T
	        dif_len = np.linalg.norm(dif)
	        if predicted > dif_len:
	          predicted = dif_len
	          predicted_class = class_index
	      if predicted_class == test_label:
	        gotIt+=1
	      else:
	        #print("failed for", predicted_class,test_label)
	        failed +=1
	        actual.append(test_label)
	        result.append(predicted_class)
	        

	    print("got", gotIt,"and failed ", failed,"accuracy = ", (gotIt /test_size)*100, "%")
	    table[mpca_index,mlda_index] = (gotIt /test_size)*100
	    mlda_index +=1
	  mpca_index+=1

print("DONE!")