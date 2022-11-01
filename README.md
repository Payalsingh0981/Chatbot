Early assessment of a disease based on
symptoms and treatment recommendations


Abstract:
 Demand for health-related information is dramatically altering
information-seeking patterns on a global scale. Individuals encounter
numerous obstacles when searching online for health-related information
on diseases, treatments, and diagnoses. People are confused due to the
vast amount of medical information available on multiple platforms. The
proposed system is a conversational agent that interacts with users to
retrieve medical information in order to provide an accurate diagnosis. For
the prevention and treatment of disease, a precise and timely evaluation
of any health-related issue is crucial. Developing a medical diagnosis
system using machine learning (ML) algorithms for the prediction of any
disease can result in a more accurate diagnosis than the conventional
method. Using multiple ML algorithms, I've developed a system for disease
prediction. The purpose of this research is to apply natural language
processing and machine learning principles to the creation of a chatbot
application. The chatbot identifies the user's symptoms, predicts the
disease, and suggests treatment through a series of questions. This system
facilitates daily health checks, informs users of their disease, and
encourages them to take the precautions necessary to remain healthy.

Introduction
Machine learning increases the speed of data processing and analysis. Using machine
learning, predictive analytics algorithms can train on even larger data sets and analyse
multiple variables in greater depth with minimal deployment modifications. Numerous
researchers have created a variety of machine learning techniques for the efficient
diagnosis of various diseases. Without being explicitly programmed, machines can
acquire knowledge through machine learning. Algorithms for machine learning can
predict and provide solutions for early disease diagnosis through the development of a
model. Detection and treatment at an early stage are the most effective means of
reducing the mortality rates associated with any disease. Therefore, the majority of
medical scientists are drawn to new predictive model technologies based on machine
learning for disease prediction. In the healthcare industry, successful decisions will be
made by extracting hidden patterns and relationships from the database. As a result of
advancements in machine learning and artificial intelligence, several classifiers and
clustering algorithms, such as K-nearest, Decision Tree, Random Forest, Support Vector
Machine (SVM), and Naive Bayes, can provide a solution to this problem.

 This paper attempts to explain how I constructed a disease prediction system based
on symptoms, treatment recommendations, and other disease-related data. Training
the prediction model using multiple algorithms concurrently was the most intriguing
aspect of this model's development. Using six distinct algorithms, including KNN,
Random Forest Classifier, SVM, Logistic Regression, Decision Tree, and Multi-layer
Perceptron classifier, I achieved a 92.42 percent accuracy rate. The concept
underlying the use of multiple algorithms is that each algorithm will attempt to
suggest a disease with greater precision, and the mode of disease prediction will
accurately identify the patient's illness.

Literature Review
1. The work proposed by (Wang X, 2008) focuses on disease prediction using clinical
data from the Presbyterian Hospital of New York. As these are clinical data,
automated disease prediction differs from text-based user input and is considerably
simpler. While matching symptom names with the user's input, the authors have
observed that users avoid using clinical terms that indicate greater complexity.
2. In the paper titled "Development of machine learning model for diagnostic disease
prediction based on laboratory tests," Dong Jin Park, Min Woo Park, Homin Lee,
Young-Jin Kim, Yeongsic Kim, and Young Hoon Park developed a novel optimised
ensemble model for disease prediction by combining a DNN (deep neural network)
model and two ML models. The optimised ensemble model had an F1-score of 81%
and a prediction accuracy of 92% for the five most prevalent diseases. Deep learning
and machine learning models differed in their prognostic accuracy and disease
categorization patterns.
3. In the paper PREDICTION OF DISEASES USING SUPERVISED LEARNING by Ashish
Kumar, Priya Ghansela, Purnima Soni, Chirag Goswami, and Parasmani Sharma, disease
predictions are extracted from a variety of sources, such as hospitals, patient discharge
slips, and the UCI repository. Afterwards, the model is trained using supervised
machine learning algorithms such as Decision Tree, Random Forest, SVM (Support
Vector Machine), and Naive-Bayes. Random Forest accuracy ranges from 54 to 95 for
SVM. The precise data source is not specified in this article.
4. In their article titled "An Overview on Disease Prediction for Preventive Care of
Health Deterioration," Mohan Kumar K N, S.Sampath, and Mohammed Imran discuss
the need for a low-cost disease prediction system capable of early disease detection. It
also includes a list of the techniques currently employed for disease prediction. Logistic
regression, Support Vector Machine (SVM), Decision Tree, and Clustering are the most
popular techniques, accounting for 27.5%, 25%, 22.5%, and 20%, respectively. Due to
the binary nature of medical data in most cases, logistic regression dominates the list.
5. The authors of the paper "Disease Prediction Using Machine Learning" by Kedar
Pingale1, Sushant Surwase2, Vaibhav Kulkarni3, Saurabh Sarage4, and Prof. Abhijeet
Karve5 discuss various algorithms that can be used for disease prediction. Existing work
focused on unstructured or textual data to improve the precision of large data sets.
Existing disease prediction methods utilised linear, KNN, and Decision Tree algorithms.
Working with different algorithms is mentioned, but implementation details are lacking.
6. In "Automated Disease Prediction System (ADPS): A User Inputbased Reliable
Architecture for Disease Prediction" by Md. Tahmid Rahman Laskar, Md. Tahmid
Hossain, Abu Raihan Mostofa Kamal, and Nafiul Rashid, the authors properly described
the algorithm's inner workings and achieved an accuracy of 83 percent, but only used
one algorithm for model building.
7. In 'PREDICTION OF DISEASES USING RANDOM FOREST CLASSIFICATION ALGORITHM',
diseases are predicted using a random forest classification algorithm. Sreevalli et al.
predicted the disease based on its symptoms using a machine-learning algorithm called
random forest. The system reduced the amount of time and money required for disease
prediction. The algorithm achieved an 84.2 percent accuracy rate.
8. In the paper by Amiya Kumar Tripathy, the need for advanced technology that
provides people with a proper healthcare management system and on which they can
rely instead of a doctor is discussed. It emphasises the need for such a system to be
portable and accurate so that people can carry it with them. The proposed system
consists of a mobile heart rate measurement that can detect heart rate and provide an
accurate diagnosis with the click of a button. In the event of an emergency, the system
also provides video conferencing to connect with a doctor. The Doc-Bot that was created
for this purpose is being converted to a mobile platform and will incorporate the concept
of symptom-based diagnosis.

Proposed Method
Our model has been trained using the supervised learning technique. Classification
algorithms are ideally suited for building disease prediction models. There are six distinct
classification algorithms that were used to construct the model.

• Logistic Regression 

• Decision tree

• Random forest

• Support Vector Machine

• Multi-layer Perceptron classifier

• K-Nearest Neighbors (KNN)

Let's examine each model in depth.

LOGISTIC REGRESSION:
• Logistic regression is a popular Machine Learning algorithm that falls under the
Supervised Learning methodology. Using a given set of independent variables, it is
employed to predict the categorical dependent variable.

• Logistic regression is used to predict the outcome of a categorical dependent
variable. The result must therefore be a categorical or discrete value. It can be either
Yes or No, 0 or 1, true or False, etc., but instead of providing the exact value as 0 or
1, probabilistic values between 0 and 1 are provided.

• Logistic Regression is very similar to Linear Regression, with the exception of their
respective applications. Linear Regression is used to solve problems involving
Regression, while Logistic Regression is used to solve problems involving
Classification. In logistic regression, rather than fitting a regression line, we fit an Sshaped logistic function that predicts two maximum values (0 or 1).

• The logistic function curve indicates the likelihood of an event, such as whether the
cells are cancerous or not, whether a mouse is obese based on its weight, etc. It is a
significant machine learning algorithm because it can classify new data using
continuous and discrete datasets and provide probabilities.

• Logistic Regression can be used to classify observations based on various types of
data and can easily identify the most effective classification.
![image](https://user-images.githubusercontent.com/82765384/199190597-8e23e39e-0fd9-4537-b74f-26cb62ae1c34.png)


DECISION TREE: 

Decision trees are useful tools, especially in situations where financial
data and outcome probability are relatively reliable. They are used to compare the costs and
likely value of potential business decision paths. They frequently involve decision
alternatives that result in multiple possible outcomes, with the probability of each outcome
being quantified. A decision tree is a branching flowchart that depicts multiple possible
decisions and outcomes. The tree begins with a node known as a decision node, which
indicates that a choice must be made. Decision trees classify instances by sorting them from
the root node to a leaf node, which provides the instance's classification. Classifying an
instance involves beginning at the tree's root node, testing the attribute specified by this node,
and then moving down the tree branch corresponding to the value of the attribute. This
procedure is then repeated for the subtree with the new node as its root.
RANDOM FOREST: Random forest is a widely used supervised machine learning algorithm
for Classification and Regression problems. It constructs decision trees from various samples
and uses the majority vote for classification and the mean for regression.

• In the random forest algorithm, n random records are selected from a data set containing k
random records.

• Each sample has its own decision tree constructed.

• Each decision tree will produce a result.

• The final output for classification and regression is determined by majority vote or average,
respectively.

SVM operates by mapping data to a high-dimensional feature space so that data points can be
categorised, even when the data are not otherwise linearly separable. A divider is identified
between the categories, and then the data are transformed so that the divider can be depicted
as a hyperplane. Then, the group to which a new record should belong can be predicted based
on the characteristics of new data. A support vector machine outputs the hyperplane (which
in two dimensions is simply a line) that best separates the tags using these data points. This
line represents the decision boundary: anything that falls on one side will be classified as
blue, while anything that falls on the other will be classified as red.
![image](https://user-images.githubusercontent.com/82765384/199190748-97d23263-bcd9-4d9b-81d0-f112862425a6.png)


In 2D, the best hyperplane is simply a line
However, what is the best hyperplane? It is the tag that maximises the margins from both
tags, according to SVM. In other words, the hyperplane (remember it's a line in this case with
the greatest distance to each tag's nearest element.
![image](https://user-images.githubusercontent.com/82765384/199190829-ed3f9c36-8420-4b3e-b61c-c000fd7e005a.png)

Not all hyperplanes are created equal.

Multilayer Perceptron Classifier:

MLP is an acronym for multi-layer perception. It is composed of densely interconnected
layers that transform any input dimension into the desired dimension. A perception with
multiple layers is a neural network with multiple layers. In order to create a neural network,
neurons are combined such that the outputs of some neurons serve as inputs for other
neurons.

A multi-layer perceptron has one input layer and one neuron (or node) for each input, one
output layer with one node for each output, and any number of hidden layers and any number
of nodes for each hidden layer. Below is a schematic representation of a Multi-Layer
Perceptron (MLP).
![image](https://user-images.githubusercontent.com/82765384/199190908-687856a3-b36f-4288-b198-82db5b8a8b38.png)


In the above multi-layer perceptron diagram, we can see that there are three inputs
and, consequently, three input nodes, as well as three hidden layer nodes. There are
two output nodes because the output layer provides two outputs. The nodes in the
input layer accept input and forward it for further processing. In the diagram, the
nodes in the input layer forward their output to each of the three nodes in the hidden
layer.

A sigmoid activation function is utilised by each node in multilayer perception. Using
the sigmoid formula, the sigmoid activation function converts real input values to
numbers between 0 and 1 for output.

(x) = 1/(1 + exp(-x))

• K-Nearest Neighbor is one of the most straightforward Machine Learning
algorithms based on the Supervised Learning technique.

• The K-NN algorithm assumes the similarity between the new case/data and
existing cases and places the new case in the category that is most similar to
existing categories.

• The K-NN algorithm stores all available data and classifies a new data point on the
basis of similarity. This implies that when new data becomes available, it can be
easily classified into a suitable category using the K- NN algorithm.

• The K-NN algorithm can be used for both Regression and Classification, but it is
typically applied to Classification problems.

• Because K-NN is a non-parametric algorithm, it makes no assumptions about the
underlying data.

It is also known as a lazy learner algorithm because it does not immediately learn
from the training set. Instead, it stores the dataset and performs an action on it at the
time of classification.

• During the training phase, the KNN algorithm simply stores the dataset, and when it
receives new data, it classifies it into a category that is highly similar to the original
category.

The K-NN algorithm can be explained using the following algorithm:

• Select the number K of the neighbours in Step 1

• Step 2: Determine the Euclidean distance between K neighbours.

• Step 3: Determine the K closest neighbours based on the Euclidean distance.

• Step 4: Count the number of data points in each category among these k
neighbours.

• Step 5: Assign the new data points to the category with the greatest number of
neighbours.

The various construction steps for this project were as follows:
1. Data Collection
2. Preprocessing of Data
3. Model Deployment
4. Disease Forecasting
Data collection:

This is the step that determines the quality of our model. The greater the quality and
quantity of data we collect, the more accurate our models' predictions will be.
A dataset for a disease prediction model was compiled from a 2004 Columbia
University study conducted at New York Presbyterian Hospital. In addition, the list of
diseases was retrieved from the Centre for Health Informatics' National Health Portal
of India (https://www.nhp.gov.in/ disease-a-z), which was developed and is
maintained by the National Health Portal of India (CHI). The script retrieves the
page's HTML code and extracts the disease list by filtering HTML tag values.
After preprocessing and multiplication, the dataset contains approximately 11000
rows with over 500 unique symptoms.

Data Preprocessing Categorical data must be converted into numeric data for
supervised learning models. The dataset we utilised consisted solely of categorical
data, as depicted in Figure. Consequently, we developed our own dataset utilising
the Columbia study database. Numerous symptoms are shared by the majority of
diseases. So we constructed a dataset with the columns of symptoms in the binary
format of 0's and 1's with respect to each disease.
![image](https://user-images.githubusercontent.com/82765384/199190968-b2631bda-793c-4468-918c-89ede400faf0.png)


Model Deployment:
![image](https://user-images.githubusercontent.com/82765384/199191022-894d9c85-9b0a-4393-b646-10dfbf1c7174.png)


Based on which model predicts diseases with the highest probability and scores, the system
prompts the user to enter symptoms. The diagram depicts the process of disease prediction
based on symptom input by the user. The subsections that follow describe each module in
detail.

The system accepts symptoms separated by commas on a single line (,). The processed
symptom list is then used for symptom expansion. Subsequently, the following
preprocessing steps are performed: split symptoms into a list based on commas, convert
symptoms to lowercase, remove stopwords, tokenize symptoms to remove punctuation
marks, and lemmatize tokens in symptoms.
![image](https://user-images.githubusercontent.com/82765384/199191077-89f6aa4e-7b42-4485-af5b-2b93efa1afe3.png)


The expanded symptom query is utilised to locate the associated symptoms within the
dataset. To locate such symptoms, each symptom from the dataset is divided into tokens,
and the presence of each token in the expanded query is examined. On the basis of this, a
similarity score is calculated, and if the symptom's score exceeds the threshold value, the
symptom qualifies as being similar to the user's symptom and is recommended to the user.
![image](https://user-images.githubusercontent.com/82765384/199191155-7695a9e9-c68b-4b50-b242-89df41ba3065.png)


The user selects one or more of the listed symptoms. On the basis of the selected
symptoms, the user is presented with additional symptoms for selection that are among the
top co-occurring symptoms with the selected symptoms. The user may choose any
symptom, skip any symptom, or cancel the symptom selection process. Figure depicts an
example of the selection and suggestion of symptoms. The final list of symptoms is then
obtained in order to compute the vector of symptoms used for prediction.
![image](https://user-images.githubusercontent.com/82765384/199191224-105d8cab-68c9-4b8e-91f2-716e30c420eb.png)
![image](https://user-images.githubusercontent.com/82765384/199191291-439f412f-9e73-4bb6-9178-c4c28bfb5ead.png)


Disease Prediction

Using the final symptom list, vectors specific to the model are computed and disease
prediction is performed. The following codes are used to predict disease using ML models.
![image](https://user-images.githubusercontent.com/82765384/199191366-b566756e-7c08-4901-9c95-a97bbcc939be.png)
 
A binary vector consisting of 1 for symptoms present in the user's selection list and 0
otherwise is computed. The dataset is used to train a machine learning model, which is then
used for prediction. The model accepts the symptom vector as input and outputs a list of
the top K diseases, ordered by decreasing individual probabilities.
![image](https://user-images.githubusercontent.com/82765384/199191438-eb376ba1-7450-4952-b470-3d6ed956ecd7.png)
![image](https://user-images.githubusercontent.com/82765384/199191642-2325610c-780c-49ac-9e49-03ec97a3ba37.png)


Then the bot displays the mode of disease predicted by all the 6 models.
![image](https://user-images.githubusercontent.com/82765384/199191797-22f050b9-b766-4cc0-a50d-45b8a054dba4.png)

In the console, the user can view the disease's specifics. The figure depicts the disease
information selected by the user. As shown in Figure, users may bypass the step by entering
"-1."
![image](https://user-images.githubusercontent.com/82765384/199191847-6f129f43-54fe-4bb0-99b1-d3542fcd4e7a.png)



Results:

The dataset is evaluated by applying multiple machine learning algorithms and comparing
the resulting accuracy. The figure depicts a comparison of various model precisions. The
algorithms with the highest accuracy are K Nearest Neighbor (92.36%) and Decision Tree
(92.36%), while Multinomial Naive Bayes has the lowest (86.13 % ).
![image](https://user-images.githubusercontent.com/82765384/199191907-9af51934-3a65-41b3-afc0-2b494a105436.png)
![image](https://user-images.githubusercontent.com/82765384/199191967-edf096df-7d33-4612-865a-039d7146163d.png)


