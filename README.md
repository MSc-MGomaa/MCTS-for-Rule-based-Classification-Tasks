# MCTS for Rule-based Classification Tasks.
To answer the question, how to generate a pattern-based classifier using Monte-Carlo Tree Search algorithm?

## Abstract
<p align="justify">
The ability to distinguish a group of samples belonging to a certain class from other samples can be achieved by creating a classifier. A classifier can be represented by a group of RuleSets, one RuleSet for each class. Each RuleSet consists of a set of rules, where each rule can be seen as a set of constraints on the features that make up the current search space.
<p align="justify">
Building a good classifier starts with finding high-quality rules from the search space, which will be used later to create the RuleSets. Current methods use beam-search, sampling, and genetic algorithms as RuleSets composition techniques. However, using such methods results in RuleSets, which have few high-quality rules. Hence the idea of using Monte-Carlo tree search algorithm, which directs the search tree to the most promising areas in the search space and, as a result, expands the rules of high quality with respect to a rule quality measure, and the class of interest.
<p align="justify">
Once the search tree is built, an external RuleSet composition technique can be used to produce the RuleSet which covers (if possible) every sample belonging to the class of interest, where the goal is to create a RuleSet consisting of high-quality rules. For this purpose, two different techniques are adopted by our method, whereby the one that works best in the given problem can be used to produce the resulting classifier. The resulting classifier can be represented by keeping all the resulting RuleSets or all except the left-out one, which is defined as the RuleSet that covers the maximum number of negative samples. In the following and through a
comprehensive set of experiments, the performance of the presented method on classification tasks was evaluated, yielding competitive predictive results.

## Important Note
The implementation of this project depends mainly on the topics mentioned in:<br>
[1 : MCTS-For-Rule-learning](https://github.com/MSc-MGomaa/MCTS-For-Rule-learning). <br>
[2 : Separate-and-Conquer-algorithm-for-pattern-set-composition](https://github.com/MSc-MGomaa/Jaccard-based-Similarity-algorithm-in-pattern-mining-tasks).<br>
[3 : Jaccard-based-Similarity-algorithm-in-pattern-mining-tasks](https://github.com/MSc-MGomaa/Jaccard-based-Similarity-algorithm-in-pattern-mining-tasks) <br>

## Classifier creation
<p align="justify">
As mentioned in the section before, for N classes, N RuleSets are generated. Now it remains, how to convert these RuleSets into a Rule-based classifier. But before getting into how the classifier is created,We’ll discuss a little bit the RuleSets that are created when using the two techniques described in the previous section.

<p align="center">
<img width="600" height="700" src="https://github.com/MSc-MGomaa/MCTS-for-Rule-based-Classification-Tasks/blob/5bc23af17421347c2cd3e9052c8fadd3872139a9/JSExample.png">

<p align="justify">
In the Figure above, it shows an example of the RuleSets created when using the Jaccard-similarity approach on the Iris dataset. Three RuleSets were generated, Iris-setosa, Iris-versicolor, and Iris-virginica as the dataset consists of only three classes. Due to the fact that S&Q and Jaccard-similarity algorithms are designed to cover every sample belonging to every label, this may result in many negative samples being covered. For example, in the same Figure, in class Iris-virginica, to cover the last sample, the Jaccard-similarity had to cover eleven negative samples, hence the need to add the constraints to control the results.


## N Classifier (Vs) N-1 Classifier
<p align="justify">
Through the resulting RuleSets, N classifier can be created by keeping all N RuleSets. But first, the RuleSets are sorted in ascending order with respect to the number of covered negative examples, Where the RuleSet with the minimum number of negative samples covered comes first. While in the second appraoch, and for retrieving a classifier, N − 1 RuleSets are only considered, where left-out RuleSet is the one that covers the max number of the negative samples.

## Classifier Evaluation
<p align="justify">
To evaluate the resulting classifier, test data is used. First, the predictions for a given test data must be generated using the resulting classifier. This can be done by exploring the test data to retrieve all the labels that make up the test data. Each sample in the test data has to pass through the RuleSets which make up the classifier. Starting with the first RuleSet, the sample has the go through each rule that makes up that RuleSet. Each rule is represented as a list of intervals, and each sample is represented as a list of numbers, where the length of the rule is equal to the length of the sample. A check is made to ensure that each value of the sample ∈ the corresponding interval of the rule. If this is the case, the sample will be labeled with the class of the current RuleSet. Otherwise, the second RuleSet will be tested.. etc. In case the current sample isn’t covered by any RuleSet of the classifier, then the left out label is assigned to that sample.

## Scikit-learn
<p align="justify">
Sklearn library offers a built in function called accuracy score, which takes the predictions of the resulting classifier, and the true values as inputs to retrieve an average accuracy value represents the current classifier. Moreover, classification report function can be used to print
detailed information about the predictions of the resulting classifier as shown in the Figure below, where the prediction accuracy is first calculated with respect to each class, like Iris-setosa = 0.93, Iris-versicolor = 0.85, and Iris-virginica = 1, and finally the mean accuracy over all predictions is calculated, as here in this case = 0.92

<p align="center">
<img width="500" height="200" src="https://github.com/MSc-MGomaa/MCTS-for-Rule-based-Classification-Tasks/blob/5bc23af17421347c2cd3e9052c8fadd3872139a9/result5.png">

## An example of the resulting classifier when using this approach
<p align="center">
<img width="500" height="200" src="https://github.com/MSc-MGomaa/MCTS-for-Rule-based-Classification-Tasks/blob/5bc23af17421347c2cd3e9052c8fadd3872139a9/JSExampleF.png">

<p align="justify">
The classifier is read as follow, if the example is not covered by Iris-virginica, or Iris-setosa, then classify it as Iris-versicolor. The examples of the RuleSets that have been removed due to the constraints, will be classified as Iris-versicolor.


## The algorithm in action
<p align="center">
<img width="800" height="600" src="https://github.com/MSc-MGomaa/MCTS-for-Rule-based-Classification-Tasks/blob/5bc23af17421347c2cd3e9052c8fadd3872139a9/result6.png">

The following configurations were used: dataset = 'Iris.csv', kFold = 10, mValues = [0.1, 0.5, 1, 5, 10, 20], jaccardValues = [0.2, 0.5, 0.8], iterationsNumber = 20, minimumSupport = 10.

4


