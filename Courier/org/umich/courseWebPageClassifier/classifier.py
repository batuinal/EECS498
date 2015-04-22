'''
Created on March 14, 2015
@author: Nikita
'''

import sys
import numpy
import math
import preprocess
import features as my_features
import tokenizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import f_regression, SelectPercentile
from sklearn import metrics

my_tokenizer = tokenizer.Tokenizer()
data_preprocessor = preprocess.Preprocess()

def main(test_dir, train_dir, test_labels_doc, train_labels_doc):
    global data_preprocessor
    train_files, train_labels = data_preprocessor.read_data(train_dir, train_labels_doc)
    test_files, test_labels = data_preprocessor.read_data(test_dir, test_labels_doc)
    tags = None
    
    features, idf, classifier = trainLogitClassifier(train_files, train_labels, 0, False, 0)
    test(test_files, test_labels, classifier, features, idf, 0, False, 0, tags)
    
    '''
    uncomment following to experiment with logit classifier with tfidf weights and non-zero df threhsold
    '''
    #features, idf, classifier = trainLogitClassifier(train_files, train_labels, 1, False, 0)
    #test(test_files, test_labels, classifier, features, idf, 1, False, 0, tags)
    
    '''
    uncomment following to experiment with structure based logit classifier
    '''
    #tags = {'a': 1, 'title': 1, 'body': 1}
    #features, idf, classifier = trainStructureBasedLogitClassifier(train_files, train_labels, 0, False, 0, tags)
    #test(test_files, test_labels, classifier, features, idf, 0, False, 0, tags)
    
    '''
    uncomment following to experiment with vocab based logit classifier
    '''
    #features, idf, classifier = trainVocabBasedLogitClassifier(train_files, train_labels, 0, False, 0)
    #test(test_files, test_labels, classifier, features, idf, 0, False, 0, tags)
    
    '''
    uncomment following to experiment with Bernoulli Naive Bayes  
    '''
    #features, idf, classifier = trainBernoulliNaiveBayes(train_files, train_labels, 2, False)
    #test(test_files, test_labels, classifier, features, idf, 2, False, 0, tags)
    
    '''
    uncomment following to experiment with Multinomial Naive Bayes  
    '''
    #features, idf, classifier = trainMultinomialNaiveBayes(train_files, train_labels, 0, False)
    #test(test_files, test_labels, classifier, features, idf, 0, False, 0, tags)
    
    '''
    uncomment following to experiment with decision trees
    '''
    #features, idf, classifier = trainDecisionTrees(train_files, train_labels, 0, False)
    #test(test_files, test_labels, classifier, features, idf, 0, False, 0, tags)
    
    '''
    uncomment following to do feature selection using f_regression
    '''
    #fTestFeatureSelection(train_files, train_labels, test_files, test_labels)
    
    '''
    uncomment following to do feature selection using document frequency
    '''
    #dfFeatureSelection(train_files, train_labels, test_files, test_labels)


def trainLogitClassifier(documents, labels, weighing_scheme, include_metadata, df_threshold):
    print 'Training'
    design_matrix, features, idf = vectorizeTrain(documents, None, weighing_scheme, include_metadata, df_threshold, None);
    classifier = LogisticRegression()
    classifier.fit(design_matrix, labels)
    return features, idf, classifier

def trainStructureBasedLogitClassifier(documents, labels, weighing_scheme, include_metadata, df_threshold, tags):
    print 'Training'
    design_matrix, features, idf = vectorizeTrain(documents, None, weighing_scheme, include_metadata, 0, tags);
    classifier = LogisticRegression()
    classifier.fit(design_matrix, labels)
    return features, idf, classifier

def trainVocabBasedLogitClassifier(documents, labels, weighing_scheme, include_metadata, df_threshold):
    print 'Training'
    design_matrix, features, idf = vectorizeTrain(documents, my_features.vocab, weighing_scheme, include_metadata, df_threshold, None);
    classifier = LogisticRegression()
    classifier.fit(design_matrix, labels)
    return features, idf, classifier

def trainBernoulliNaiveBayes(documents, labels, weighing_scheme, include_metadata):
    print 'Training'
    design_matrix, features, idf = vectorizeTrain(documents, None, weighing_scheme, include_metadata, 0, None);
    classifier = BernoulliNB()
    classifier.fit(design_matrix, labels)
    return features, idf, classifier

def trainMultinomialNaiveBayes(documents, labels, weighing_scheme, include_metadata):
    print 'Training'
    design_matrix, features, idf = vectorizeTrain(documents, None, weighing_scheme, include_metadata, 0, None);
    classifier = MultinomialNB()
    classifier.fit(design_matrix, labels)
    return features, idf, classifier

def trainDecisionTrees(documents, labels, weighing_scheme, include_metadata):
    print 'Training'
    design_matrix, features, idf = vectorizeTrain(documents, None, weighing_scheme, include_metadata, 0, None);
    classifier = DecisionTreeClassifier()
    classifier.fit(design_matrix, labels)
    return features, idf, classifier

'''
Feature selection experiment based on univariate linear regression test values
'''
def fTestFeatureSelection(train_files, train_labels, test_files, test_labels):
    design_matrix, features, _ = vectorizeTrain(train_files, None, 0, False, 0, None)
    classifier = LogisticRegression()
    for p in range(10):
        percentile = 100-p*10
        print 'Selecting {0}% of features'.format(percentile)
        feat_sel = SelectPercentile(f_regression, percentile)
        X_sel = feat_sel.fit_transform(design_matrix, train_labels)
        f_inds = feat_sel.get_support(indices=True)
        print 'Using {0} features'.format(len(f_inds))
        classifier.fit(X_sel, train_labels)
        test(test_files, test_labels, classifier, [features[d] for d in f_inds], None, 0, False, 0, None)

'''
Feature selection experiment based on document frequency thresholding
'''        
def dfFeatureSelection(train_files, train_labels, test_files, test_labels):
    for p in range(8):
        doc_threshold = 5 * p
        print 'Setting threshold doc frequency to {0} of features'.format(doc_threshold)
        features, idf, classifier = trainLogitClassifier(train_files, train_labels, 0, False, doc_threshold)
        print 'Using {0} features'.format(len(features))
        test(test_files, test_labels, classifier, features, idf, 0, False, 0, None)        

def test(documents, labels, classifier, features, idf, weighing_scheme, include_metadata, df_threshold, tags):
    print 'Testing'
    design_matrix = vectorizeTest(documents, features, idf, weighing_scheme, include_metadata, df_threshold, tags)
    predicted = classifier.predict(design_matrix)
    f1_score = metrics.f1_score(labels, predicted, average='micro')
    accuracy = metrics.accuracy_score(labels, predicted)
    print accuracy, f1_score
    return accuracy, f1_score

'''
vectorizing train documents
param: Features None or a list in case of domain specific vocab
weighing scheme: 0 for df, 1 for tfidf and 2 for bernoulli
include metadata: boolean flag to include metadata features
df_threshold: threshold on document frequency
tags: if only specific tags should be vectorized
'''
def vectorizeTrain(documents, features, weighing_scheme, include_metadata, df_threshold, tags):
    global data_preprocessor
    print 'Preprocessing'
    inverted_index = None
    if tags is None:
        inverted_index = data_preprocessor.indexDocuments(documents, include_metadata)
    else:
        inverted_index = data_preprocessor.indexDocumentsForTags(documents, tags) 
    if features is None:
        features = getFeatures(inverted_index, df_threshold)
    design_matrix = None
    idf = None
    if weighing_scheme == 0:
        design_matrix = getDFDesignMatrix(inverted_index, features, documents)
    elif weighing_scheme == 1:
        idf = {}
        for term in features:
            if term in my_features.metadataKeys:
                idf[term] = 1
            else:
                idf[term] = data_preprocessor.inverseDocumentFrequency(term, inverted_index, len(documents))
        design_matrix = getTFIDFDesignMatrix(inverted_index, features, idf, documents)
    else:
        design_matrix = getBernoulliDesignMatrix(inverted_index, features, documents)
    return design_matrix, features, idf

'''
vectorizing test documents
param: Features learned from training
inverse document frequencies learned from training
weighing scheme: 0 for df, 1 for tfidf and 2 for bernoulli
include metadata: boolean flag to include metadata features
df_threshold: threshold on document frequency
tags: if only specific tags should be vectorized
'''
def vectorizeTest(documents, features, idf, weighing_scheme, include_metadata, df_threshold, tags):
    global data_preprocessor
    if weighing_scheme == 1:
        design_matrix = numpy.zeros((len(documents), len(features))) 
        docids = [data_preprocessor.getDocId(d) for d in documents]
        for docIndex in range(len(documents)):
            document = documents[docIndex]
            doc_id = docids[docIndex]
            inverted_index = None
            if tags is None:
                inverted_index = data_preprocessor.indexDocuments([document], include_metadata)
            else:
                inverted_index = data_preprocessor.indexDocumentsForTags([document], tags) 
            document_vector = numpy.zeros((1, len(features)))
            document_length = 0.0
            for termIndex in range(len(features)):
                term = features[termIndex]
                tfidf = data_preprocessor.term_frequency(term, doc_id, inverted_index) * 1.0 * idf.get(term, 0)
                document_length = document_length + (tfidf ** 2)
                document_vector[0][termIndex] = tfidf
            document_length = math.sqrt(document_length)
            document_vector /= document_length
            design_matrix[docIndex] = document_vector
        return design_matrix    
    else:
        design_matrix,_,_ = vectorizeTrain(documents, features, weighing_scheme, include_metadata, df_threshold, tags)
        return design_matrix 

'''
design matrix for bernoulli - features are either absent or present
'''
def getBernoulliDesignMatrix(inverted_index, features, documents):
    design_matrix = numpy.zeros((len(documents), len(features)))
    docids = [data_preprocessor.getDocId(d) for d in documents]
    for termIndex in range(len(features)):
        term = features[termIndex]
        docs = inverted_index[term]
        for doc in docs:
            design_matrix[docids.index(doc)][termIndex] = 1
    return design_matrix

'''
df vectorizer
'''
def getDFDesignMatrix(inverted_index, features, documents):
    design_matrix = numpy.zeros((len(documents), len(features)))
    docids = [data_preprocessor.getDocId(d) for d in documents]
    for termIndex in range(len(features)):
        term = features[termIndex]
        docs = inverted_index[term]
        for doc in docs:
            design_matrix[docids.index(doc)][termIndex] = inverted_index[term][doc]
    return design_matrix

'''
tfidf vectorizer
'''
def getTFIDFDesignMatrix(inverted_index, features, inverseDocumentFrequencies, documents):
    global data_preprocessor
    design_matrix = numpy.zeros((len(documents), len(features)))
    docids = [data_preprocessor.getDocId(d) for d in documents]
    document_lengths = [data_preprocessor.getDocumentLength(doc_id, inverted_index, len(documents)) for doc_id in docids]
    for termIndex in range(len(features)):
        term = features[termIndex]
        docs = inverted_index[term]
        for doc in docs:
            doc_index = docids.index(doc)
            doc_length = document_lengths[doc_index]
            design_matrix[doc_index][termIndex] = data_preprocessor. term_frequency(term, doc, inverted_index) * 1.0 * inverseDocumentFrequencies.get(term, 0) / doc_length
    return design_matrix

'''
extracts features from an inverted index. 
A feature is included if the document frequency exceeds the threshold
'''    
def getFeatures(inverted_index, threshold):
    features_list = []
    for term in inverted_index.keys():
        document_frequency = len(inverted_index[term])
        if document_frequency > threshold:
            features_list.append(term)
    return features_list

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
