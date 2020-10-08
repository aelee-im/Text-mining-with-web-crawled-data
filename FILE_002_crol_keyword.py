from bs4 import BeautifulSoup
import time
import pandas as pd
import urllib.request

db_files = pd.read_csv('sample_url.csv', header=None)
db_files.columns = ['urls']
    
n=0 
df_db = []
db_list = []
        
for i in db_files['urls']:

        
    t1 = time.time()
    
    n += 1    
  
    try:
        
        with urllib.request.urlopen(i) as url:
            url = url.read()
            
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

    #lowercase for text1
    import re
    text1=text1.lower()
    text1_token = re.findall(r'\b[a-z]{3,15}\b', text1)

    ####  remove stopwords

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    text1_clean = [w for w in text1_token if not w in stop_words]
    
    
    # Lemmatize list of words and join
    from nltk.stem import WordNetLemmatizer
    lemma = WordNetLemmatizer()
    text1_lemma= [lemma.lemmatize(x) for x in text1_clean]


    #topwords
    from nltk import FreqDist
    fdist=FreqDist(text1_lemma)
    tops=fdist.most_common(10)
    table=""
    for key in tops:
        table +='\''+key[0]+'\''
        table +=','

    print(table)

    db_list.append(table)       
        


            
