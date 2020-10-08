import nltk
import string
import itertools
import gensim
import math
from sklearn.feature_extraction.text import TfidfVectorizer
#import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from pandas import DataFrame
import urllib.request


db_files = pd.read_csv('keyphrases_compare.csv')
  
n=0 
df_db = []
        
for i in db_files['urls']:

        
    t1 = time.time()
    
    n += 1    
        
    try:
        url = urllib.request.urlopen(i).read()
        soup = BeautifulSoup(url)

        
    # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out
    except:
        pass

    
    # get text
    text_n = soup.get_text()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text_n.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text1 = '\n'.join(chunk for chunk in chunks if chunk)    


    t2 = time.time()
   
    print(n, i, ", ", t2-t1)


    
    def read_txt(text_files):

        doc = text_files.replace('\\n', ' ')
        doc = doc.replace('\\', ' ')
            
        return doc
    
    def extract_chunks(text_string,max_words=3,lemmatize=False):
    
        # Any number of adjectives followed by any number of nouns and (optionally) again
        # any number of adjectives folowerd by any number of nouns
        grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    
        # Makes chunks using grammar regex
        chunker = nltk.RegexpParser(grammar)
        
        # Get grammatical functions of words
        # What this is doing: tag(sentence -> words)
    
        tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text_string))
        
        # Make chunks from the sentences, using grammar. Output in IOB.
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                            for tagged_sent in tagged_sents))
    
        # Join phrases based on IOB syntax.
        candidates = [' '.join(w[0] for w in group).lower() for key, group in itertools.groupby(all_chunks, lambda l: l[2] != 'O') if key]
        
        # Filter by maximum keyphrase length
        candidates = list(filter(lambda l: 2<= len(l.split())<=3, candidates))
        
        # Filter phrases consisting of punctuation or stopwords
        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        candidates = list(filter(lambda l: l not in stop_words and not all(c in punct for c in l),candidates))
        
        # lemmatize
        if lemmatize:
            lemmatizer = nltk.stem.WordNetLemmatizer().lemmatize
            candidates =  [lemmatizer(x) for x in candidates]
    
        return candidates
    
    def extract_terms_with_corpus_sklearn(text_files, number_of_terms=10, max_features=20, max_words=3, lemmatize=True, train_on_script = True):
    
        # tokenizer
        analyzer = lambda s: extract_chunks(read_txt(s),lemmatize=lemmatize,max_words=max_words)
    
        # All-in-one object for tfidf calculation
        tfidf_vectorizer = TfidfVectorizer(input=text_files, analyzer = analyzer, max_features=max_features)
        
        # fit training data & get tfidf matrix
        if train_on_script:
            tfidf_mat = tfidf_vectorizer.fit(text_files[0:])
        else: 
            tfidf_mat = tfidf_vectorizer.fit(text_files[1:])
        
        # transform first file
        tfidf_script = tfidf_vectorizer.transform([text_files[0]])
        
        # get map between id and term
        id2term = tfidf_vectorizer.get_feature_names()
    
        return [(id2term[i],tfidf_script[0,i]) for i in tfidf_script.toarray()[0,:].argsort()[::-1][0:number_of_terms]]
    
    def kpex(text_files,package='sklearn',number_of_terms=10,max_features=20,max_words=3,lemmatize=True,train_on_script=True):

        if number_of_terms>max_features:
            raise Exception("number_of_terms has to be smaller than max_features")
        if (package is not 'sklearn') and (package is not 'gensim'):
            raise Exception("Package should be sklearn or gensim")
    
        if package == 'sklearn':
            return extract_terms_with_corpus_sklearn(text_files,number_of_terms=10,max_features=20,max_words=3,lemmatize=True,train_on_script=True)
        else:
            return extract_terms_with_corpus_gensim(text_files,number_of_terms=number_of_terms,max_words=max_words,lemmatize=lemmatize)
     
    def extract_terms_with_corpus_gensim(text_files,number_of_terms=10,max_words=3, lemmatize=True):
       
        # Make chunks 
        chunked_texts = [extract_chunks(read_txt(text),max_words=max_words,lemmatize=lemmatize) for text in text_files]
       
        # Mapping between id and term 
        dictionary = gensim.corpora.Dictionary(chunked_texts)
        
        # Bag of words representation of the text
        cp = [dictionary.doc2bow(boc_text) for boc_text in chunked_texts]
        
        # tf/idf frequency model
        tf = gensim.models.TfidfModel(cp[0:],normalize=False,wglobal = lambda df,D: math.log((1+D)/(1+df))+1)
        
        # transform script
        ts = tf[cp][0]
        
        # sort by score
        s = sorted(ts,key=lambda ts: ts[1], reverse=True)
        
        # retranslate in terms
        terms = [(dictionary[s[0]],s[1]) for s in s]
    
        return terms[0:number_of_terms]

    text_files = text1.splitlines()
    terms = kpex(text_files)
    df_terms = DataFrame(list(terms), columns=["keyphrases", "TF-IDF"])
    df_phrases = df_terms['keyphrases'].head(5)

    
    print(df_terms)

    df_db.append(df_phrases)    
            
    # only if you want to export output to csv files
# =============================================================================
#     import csv
#     
#     f = open('keyphrases_new_103.csv', 'w')
#     w = csv.writer(f, delimiter = ';')
#     w.writerows(df_db)
#     f.close()
# =============================================================================
 