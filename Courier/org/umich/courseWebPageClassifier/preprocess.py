'''
Created on March 14, 2015
@author: Nikita
'''

from collections import defaultdict
import os
import tokenizer
import numpy
import re
import math

class Preprocess(object):
    def __init__(self):
        self.myTokenizer = tokenizer.Tokenizer()
    '''
    reads files names from a given directory into an array and corresponding labels from a given label file 
    '''
    def read_data(self, dirname, label_file):
        filenames = []
        for path, _, files in os.walk(dirname):
            filenames += [os.path.join(path, filename) for filename in files]
        filenames = sorted(filenames)
        labels = numpy.genfromtxt(fname=label_file, skip_header=1, delimiter=',', usecols=(1), converters={1:lambda s: 1 if s == '1' else -1})
        return numpy.array(filenames), labels
    
    '''
    helper function to extract doc id from file name
    '''
    def getDocId(self, doc_file_name):
        mid = re.sub('^[A-Za-z\-]+', '', os.path.basename(doc_file_name)).lstrip('0')
        return int(re.sub('.html$', '', mid))
    
    '''
    Creates an inverted index from a set of documents
    '''
    def indexDocuments(self, documents, include_metadata):
        inverted_index = defaultdict(dict)
        for data_file in documents:
            with open(data_file) as f:
                docId = self.getDocId(data_file)
                doc_content = f.read()
                self.indexDocument(doc_content, docId, inverted_index, include_metadata)  
        return inverted_index
    
    '''
    Given a map of tags to weights, indexes the documents content embedded in the tags while using appropriate weights
    '''
    def indexDocumentsForTags(self, documents, tags):
        inverted_index = defaultdict(dict)
        for data_file in documents:
            with open(data_file) as f:
                docId = self.getDocId(data_file)
                doc_content = f.read()
                for tag in tags.keys():
                    self.indexDocumentForTag(doc_content, docId, inverted_index, tag, tags[tag])
        return inverted_index           
    
    '''
    Tokenizes the document content embedded in the specific tag and updates the inverted index
    Optionally accepts a weight to tune the frequency of the tokens derived from the tag 
    '''            
    def indexDocumentForTag(self, document_content, doc_id, inverted_index, tag, weight): 
        terms = self.myTokenizer.getTokensForTag(document_content, tag)
        vocab_terms = set(terms)
        for term in vocab_terms:
            term_frequency = terms.count(term)
            original_weight = 0
            if term in inverted_index:
                original_weight = inverted_index[term].get(doc_id, 0)
            inverted_index[term][doc_id] = original_weight + (weight * term_frequency)
        return inverted_index

    '''
    Tokenizes the document content and updates the inverted index
    Boolean flag to include special features
    '''  
    def indexDocument(self, document_content, doc_id, inverted_index, include_metadata):  
        metadata = self.myTokenizer.getMetaData(document_content)
        terms = self.myTokenizer.getTokens(document_content)
        vocab_terms = set(terms)
        for term in vocab_terms:
            term_frequency = terms.count(term)
            inverted_index[term][doc_id] = term_frequency
        if include_metadata:
            for entry in metadata.keys(): 
                inverted_index[entry][doc_id] = metadata[entry]
        return inverted_index

    '''
    Inverse Document Frequency of a term given an inverted index and size of collection
    '''
    def inverseDocumentFrequency(self, term, inverted_index, collection_size):
        if term in inverted_index.keys():
            document_frequency = len(inverted_index[term]) 
            if document_frequency > 0:
                return math.log10(collection_size * 1.0/document_frequency)
        return 0.0

    '''
    Frequency of a term in an inverted_index
    '''
    def term_frequency(self, term, doc_id, inverted_index):
        if doc_id in inverted_index[term]:
            return inverted_index[term][doc_id]
        return 0

    
    '''
    Normalization factor for tfidf - document length
    '''
    def getDocumentLength(self, doc_id, inverted_index, collection_size):
        l = 0.0
        for term in inverted_index.keys():
            if doc_id in inverted_index[term]:
                tf = inverted_index[term][doc_id]
                document_frequency = len(inverted_index[term]) 
                idf = 0.0
                if document_frequency > 0:
                    idf = math.log10(collection_size * 1.0/document_frequency)
                l = l + (tf * 1.0 * idf)**2
        return math.sqrt(l)
    
        
