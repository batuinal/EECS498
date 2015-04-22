'''
Created on March 14, 2015
@author: Nikita
'''

vocab = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'course', 'class', 'syllabus', 'handout', 'homework', 'lecture', 'notes', 'slides', 'solution', 'problem', 'program', 'instructor', 'information', 'project', 'paper', 'guide', 'study', 'activities', 'projects', 'professor', 'office']
relevantURLsPattern = ['slide', 'handout', 'schedule', 'syllabus', 'homework', 'lecture', 'assignment', 'project', 'exam', 'midterm', 'final', 'notes', 'staff', 'hours', 'course-info', 'piazza']
relevantTags = ['li', 'ul', 'a', 'h1', 'h2', 'h3'];
stopWords = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'few', 'from', 'for', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on', 'or', 'our', 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that', 'this', 'to', 'us', 'was', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you', 'your']
numericCharacterCount = 'numericCharCount'
relevantURLsCount = 'URLCount'
relevantTagsCount = 'TagCount'
documentLength = 'docLength'
metadataKeys = [relevantURLsCount, relevantTagsCount, numericCharacterCount, documentLength]