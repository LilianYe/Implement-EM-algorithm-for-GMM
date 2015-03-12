import os.path
import random
from numpy import *
import math
import matplotlib.pyplot as plt

def get_valid_filename(msg):
    """ (str) -> str

    Prompt the user, using msg, to type the name of a file. This file should 
    exist in the same directory as the starter code. If the file does not
    exist, keep re-prompting until they give a valid filename.
    Return the name of that file.
    >> "d:\\temp\\train.txt"
    """

    filename = input(msg)
    # use os.path.isfile instead of os.path.exists
    # to make sure it's a file rather than a directory
    while not os.path.isfile(filename):
        print("That file does not exist.")
        filename = input(msg)

    return filename

def read_file(filename):
    """ (str) -> list

    Read data from filename and return it as 
    a list of features. 
    """
    
    file = open(filename, 'r')

    result = []
    # Read each remaining feature and convert each one to float.
    for line in file:
        record = line.strip()
        record_float = [float(c) for c in record.split()]
        result.append(record_float)
    file.close()

    return result   


def mul_norm(x, miu, cov):
    """(array, array, array) -> float

    return the probability the date x is produced by a 
    Gaussian components, the miu and cov are the parameters 
    of the Gaussian distribution.
    pay attention: k=2
    >>> mul_norm(array([0.5, 0.5]), array([0, 0]), array([[1,-1], [1,1]])) 
    0.12394999430965298
    """
    result = math.pow(linalg.det(cov), -0.5) / (2 * math.pi)
    temp = x-miu
    result *= exp(-0.5 * dot(dot(temp, linalg.inv(cov)), temp.T))
    return result

def compute_s(c, r, miu, cov, x):
    """(list, list, array, array, array) -> float

    Return the probability of the data produced by a GMM.
    c, miu, cov are the parameters of the GMM
    """

    result = 0.0
    for k in range(len(c)):
        temp = x - miu[k]
        result += r[k] * math.log(c[k]) - 0.5 * r[k] * (math.log(linalg.det(cov[k])) +  dot(dot(temp, linalg.inv(cov[k])), temp.T))
    return result


def compute_L(c, r, miu, cov, tag_array):
    """(list, list, array, array, array) -> float

    Return the value of the likelihood function given parameters and data
    """

    result = 0.0
    for i in range(len(tag_array)):
        result += compute_s(c, r[i], miu, cov, tag_array[i])

    return result


def update_pram(c, r, miu, cov, tag_array):
    """(list, list, list(array), list(array), array) -> none

    Update the parameters of the GMM according the rules
    """

    r_sum = []
    for i in range(len(tag_array)):
        temp = 0.0
        for k in range(len(c)):
            temp += c[k] * mul_norm(tag_array[i], miu[k], cov[k])
        r_sum.append(temp)

    c_new = [0.0] * len(c)
    for i in range(len(tag_array)):
        for k in range(len(c)):
            r[i][k] = c[k] * mul_norm(tag_array[i], miu[k], cov[k]) / r_sum[i]
            c_new[k] += r[i][k]

    c = c_new

    for k in range(len(c)):
        tmp = array(r)[:,k]
        miu[k]= sum(array([tmp]).T * tag_array, axis=0) / c[k]
        cov[k] = dot(dot((tag_array - miu[k]).T, diag(tmp)), (tag_array-miu[k])) / c[k]

    c = [t/sum(c) for t in c]

def compute_r(c, miu, cov, tarray):
    """(list, list(array),list(array),array) -> list
    
    Return the r of dev data or test data
    r[n][m] is the poseterior probability of tarray[n] 
    belonging to component m
    """

    t_r = [[0.25 for col in range(len(c))] for row in range(len(tarray))]

    tr_sum = []
    for i in range(len(tarray)):
       temp = 0.0
       for k in range(len(c)):
          temp += c[k] * mul_norm(tarray[i], miu[k], cov[k])
       tr_sum.append(temp)

    for i in range(len(tarray)):
       for k in range(len(c)):
          t_r[i][k] = c[k] * mul_norm(tarray[i], miu[k], cov[k]) / tr_sum[i]
    return t_r

def train_module(miu, covs, c, tag_array):
    """(array, array) -> [list, list(array), list(array)]

    Train the GMM module given the tag_array as training data
    miu is the initial parameter of the GMM 
    other parameters of GMM are initialized automatically with
    the info of tag_array
    """
    #r = [[1.0/len(miu) for col in range(len(miu))] for row in range(len(tag_array))] 
    r = compute_r(c, miu, covs, tag_array)
    L_list = []
    L0 = compute_L(c, r, miu, covs, tag_array)
    print L0
    L_max = L0
    c_max = c
    miu_max = miu
    covs_max = covs

    L_list.append(L0)
    update_pram(c, r, miu, covs, tag_array)
    L1 = compute_L(c, r, miu, covs, tag_array)

    L_list.append(L1)
    
    while abs((L1 - L0)/L0)> 1.0e-6:
        update_pram(c, r, miu, covs, tag_array)
        L0 = L1
        if L1 > L_max:
            c_max = c
            miu_max = miu
            covs_max = covs
        L1 = compute_L(c, r, miu, covs, tag_array)
        
        print L1
        L_list.append(L1)

    return c_max, miu_max, covs_max, L_list

# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2-vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i] = dataSet[index]
    return centroids


# K-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = zeros((numSamples,2))
    clusterChanged = True

    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        for i in range(numSamples):
            minDist = 10000000.0
            minIndex = 0

            # find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j], dataSet[i])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # update its cluster
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i] = minIndex, minDist**2

        # update centroids
        for j in range(k):
            pointsInCluster = dataSet[clusterAssment[:,0] == j]
            centroids[j] = mean(pointsInCluster, axis = 0)

    # get the initialization for c and covs
    c = []
    covs = []
    #colors = ['b', 'r', 'g', 'k']
    for i in range(k):
        pointsInCluster = dataSet[clusterAssment[:,0] == i]
        c.append(len(pointsInCluster) * 1.0 / len(dataSet))
        covs.append(cov(pointsInCluster.T))
        #plt.scatter(pointsInCluster[:,0],pointsInCluster[:,1],c=colors[i])
    
    #plt.show()
        
    return centroids, c, covs

def kmeans_best(dataSet, k, n):
    best_miu, best_c, best_covs = kmeans(dataSet, k)
    best_r = compute_r(best_c, best_miu, best_covs, dataSet)
    best_L = compute_L(best_c, best_r, best_miu, best_covs, dataSet)    
    
    for i in range(1,n):
        temp_miu, temp_c, temp_covs = kmeans(dataSet, k)
        temp_r = compute_r(temp_c, temp_miu, temp_covs, dataSet)
        temp_L = compute_L(temp_c, temp_r, temp_miu, temp_covs, dataSet)   
        if temp_L > best_L:
            best_miu, best_c, best_covs = temp_miu, temp_c, temp_covs
        print best_L
        
    return best_miu, best_c, best_covs


# #############################
# The main program begins here
# #############################


if __name__ == '__main__':

    prompt = 'Enter the name of the train file: '
    train_filename = get_valid_filename(prompt)
    #train_filename = "d:\\tmp\\train.txt"
    train_list = read_file(train_filename)

    prompt = 'Enter the name of the dev file: '
    dev_filename = get_valid_filename(prompt)
    #dev_filename = "d:\\tmp\\dev.txt"
    dev_list = read_file(dev_filename)

    prompt = 'Enter the name of the test file: '
    test_filename = get_valid_filename(prompt)
    #test_filename = "d:\\tmp\\test.txt"
    test_list = read_file(test_filename)

    dev_array = array(dev_list)
    tag1_list = []
    tag2_list = []
    dev_tag = dev_array[:,-1]
    dev_array = dev_array[:,:-1]
    test_array = array(test_list)
    test_tag = []

    for line in train_list:
        if line[-1] == 1:
            tag1_list.append(line[:-1])
        else:
            tag2_list.append(line[:-1])
    tag1_array = array(tag1_list)
    tag2_array = array(tag2_list)

    
    # train the GMM
    # uncomment this part if you want to train the model
    """
    k_dev = 4
    n_dev = 5
    miu_1, c_1, cov_1 = kmeans_best(tag1_array, k_dev, n_dev)
    miu_2, c_2, cov_2 = kmeans_best(tag2_array, k_dev, n_dev)
    c1, miu1, cov1, y_list = train_module(miu_1, cov_1, c_1, tag1_array)
    c2, miu2, cov2, y_list = train_module(miu_2, cov_2, c_2, tag2_array)
    """

    # train results 
    # uncomment this part if you want to use the training result directly 
    """
    c1 = [0.25, 0.25, 0.24916666666666668, 0.25083333333333335]
    c2 = [0.24958333333333332, 0.25125, 0.2520833333333333, 0.24708333333333332]
    miu1 = array([[ 1.99554815,  1.99310704],
       [ 3.00306266, -0.01793299],
       [ 0.99800061, -0.99400902],
       [ 0.49893327,  0.50927098]])
    miu2 = array([[ -5.03758514e-01,  -1.49810706e-02],
       [  7.94691979e-01,   1.50772274e+00],
       [  2.50372184e+00,   9.86791679e-01],
       [  1.49905249e+00,  -8.21893912e-05]])
    cov1 = [array([[ 0.07076093, -0.00235192],
       [-0.00235192,  0.07503675]]), array([[ 0.08503751, -0.00261252],
       [-0.00261252,  0.05237745]]), array([[ 0.04838777,  0.00073051],
       [ 0.00073051,  0.08075886]]), array([[ 0.03790203, -0.00265137],
       [-0.00265137,  0.04073969]])]
    cov2 = [array([[ 0.06792405, -0.00594001],
       [-0.00594001,  0.06910319]]), array([[ 0.0325016 ,  0.00038503],
       [ 0.00038503,  0.04758186]]), array([[ 0.03950061,  0.00030938],
       [ 0.00030938,  0.05478033]]), array([[ 0.0727013 , -0.00493445],
       [-0.00493445,  0.07491689]])]
    """
    
    # test on dev datasets
    # uncomment this part if you want to test 
    """
    dev_r1 = compute_r(c1, miu1, cov1, dev_array)
    dev_r2 = compute_r(c2, miu2, cov2, dev_array)
    right_num = 0
    for i in range(len(dev_array)):
        if compute_s(c1, dev_r1[i], miu1, cov1, dev_array[i]) > compute_s(c2, dev_r2[i], miu2, cov2, dev_array[i]):
            if dev_tag[i] == 1:
                right_num += 1
        else:
            if dev_tag[i] == 2:
                right_num += 1

    print right_num
    """

    # output results on test datasets
    """
    test_r1 = compute_r(c1, miu1, cov1, test_array)
    test_r2 = compute_r(c2, miu2, cov2, test_array)

    for i in range(len(test_array)):
      if compute_s(c1, test_r1[i], miu1, cov1, test_array[i]) > compute_s(c2, test_r2[i], miu2, cov2, test_array[i]):
        test_tag.append(1)
      else:
        test_tag.append(2)

    test_result = open("d:\\tmp\\test_result.txt", 'w')

    for i in range(len(test_array)):
        t_list = test_list[i] + [test_tag[i]]
        t_list = [str(t) for t in t_list]
        test_result.write(" ".join(t_list) + "\n")
    test_result.close()
    """

    print "finish"   