********************
System Prerequisites: 

Crawling: Crawling requires an external python library Scrapy
To install Scrapy use: pip install Scrapy
Further instructions can be found here: http://doc.scrapy.org/en/latest/intro/install.html

To put our spider to work, go to the courseWebPageCrawler directory and run:
scrapy crawl courseWebPageCrawler

Classifier: Classifer requires Scikit-Learn python library

Scikit is preinstalled on CAEN. To use the package directory run:
module load python

To install Scikit use:
pip install -U scikit-learn
Further instructions can be found here: http://scikit-learn.org/stable/install.html

********************
The classifier takes the following arguments:
(1) Test Directory - Path to directory containing test files
(2) Train Directory - Path to directory containing training files
(3) Test Labels - Path to file containing labels for test files
(4) Train Labels - Path to file containing labels for training files

Default Classifier Configuration:
The classifier by default uses Logistic Regression. Bag-of-words with term frequencies as features of a document. 
Metadata features are excluded and no feature selection techniques are employed. 

The classifier can be run in following configurations:
(1) Logistic Regression with different weighing schemes (0=term frequencies, 1=tfidf, 2=boolean)
(2) Different classification methods - Bernoulli Naive Bayes, Multinomial Naive Bayes, Logistic Regression, Decision Trees
(3) Feature selection - based on document frequencies and F1 regression test values for features
(4) Feature design - special features from metadata like relative number of relevant tags, relative number of relevant urls, number of numeral characters. Relevant tag and url patterns are specified in features.py
(5) Restricted vocab - Uses the vocab list from features.py
(6) Differential weighing of tags. Uses different weighing schemes for different sources of information.

Methods corresponding to each of these configurations are included in classifier.py

********************
