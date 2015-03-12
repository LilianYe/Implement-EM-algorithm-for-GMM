Implement EM algorithm for GMM

This is a binary classification task. 
Features used are 2-dimensional feature.
You will use Gaussian Mixture Model to accomplish the task.

Data Description

-------------------------------------------------------------------------------

train.txt         Training data. 

dev.txt           Development data for tuning and self testing. 

test.txt          Evaluation data for testing the model. Only 


Data Format
-------------------------------------------------------------------------------

Each line of the data file represent a sample in the below format:

Feature-Dim1   Feature-Dim2   Class-Label

Note that Both features and labels are given for train.txt and dev.txt. 
Only features are given for test.txt.

Requirement

-------------------------------------------------------------------------------

1. Implement training and testing algorithms for GMM. 
   Programmes must be written in C/C++ or python. Matlab is NOT allowed.


2. Use train.txt for training and check the result on dev.txt.
   The complexity of GMM and initialisation of GMM will be decided by you.


3. Once the final GMM configuration is fixed, you will perform classification
   on test.txt and save the result in the same format as dev.txt.


4. Final submission should include:
   
a. Detailed report including:
      
i. Initialisation of GMM
      
ii. GMM parameter tuning process (likelihood change, result on dev.txt etc.)
      
iii. Analysis and discussion 
   
b. Classification result: test.txt with label
   
c. Source code which can be compiled and/or run under 64-bit linux machine (Ubuntu)

 
