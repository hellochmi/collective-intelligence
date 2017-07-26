# docclass.py
# a program for classifying documents

import re
import math
import sqlite3
from pysqlite2 import dbapi2 as sqlite

# method to automate dumping of training data
def sampletrain(cl):
    cl.train('nobody owns the water.','good')
    cl.train('the quick rabbit jumps fences','good')
    cl.train('buy pharmaceuticals now','bad')
    cl.train('make quick money at the online casino','bad')
    cl.train('the quick brown fox jumps','good')

# extracts individual words from text and returns
# a set of unique words only

def getwords(doc):
    # create new regex object 'splitter' representing non-alpha character
    splitter = re.compile('\\W*')
    # create list of lowercase words split by splitter in doc
    words = [s.lower() for s in splitter.split(doc)
             # but only if the word is greater than 2 characters and less than 20
             if len(s)>2 and len(s)<20]
    # return a dict of unique words only
    return dict([(w,1) for w in words])

# three instance variables are fc, cc, and getfeatures
# fc will store the counts for different features in different classifications
# cc stores the count of how many times every classification has been used,
# needed for probability calculations
# getfeatures will be used to extract features from items being classified
# (in this case, our getwords function)

class classifier:
    
    def __init__(self,getfeatures,filename=None):
        # Counts of feature/category combinations
        self.fc={}
        # Counts of documents in each category
        self.cc={}
        self.getfeatures=getfeatures
        self.thresholds={}

    def setdb(self,dbfile):
        self.con=sqlite.connect(dbfile)
        self.con.execute('create table if not exists fc(feature,category,count)')
        self.con.execute('create table if not exists cc(category,count)')

    def setthreshold(self,cat,t):
        self.thresholds[cat]=t

    def getthreshold(self,cat):
        if cat not in self.thresholds: return 1.0
        return self.thresholds[cat]

    # increase count of feature/category pair
    def incf(self,f,cat):
        count=self.fcount(f,cat)
        if count==0:
            self.con.execute("insert into fc values ('%s','%s',1)"
                             % (f,cat))
        else:
            self.con.execute(
                "update fc set count=%d where feature='%s' and category='%s'"
                % (count+1,f,cat))

    # increase the count of a category
    def incc(self,cat):
        count=self.catcount(cat)
        if count == 0:
            self.con.execute("insert into cc values ('%s',1)" % (cat))
        else:
            self.con.execute("update cc set count=%d where category='%s'" % (count+1,cat))

    # the number of times a feature has appeared in a category
    def fcount(self,f,cat):
        res=self.con.execute(
            'select count from fc where feature="%s" and category="%s"'
            %(f,cat)).fetchone()
        if res==None: return 0
        else: return float(res[0])

    # the number of items in a category
    def catcount(self,cat):
        res=self.con.execute('select count from cc where category="%s"'
                             %(cat)).fetchone()
        if res==None: return 0
        else: return float(res[0])
        

    # the total number of items
    def totalcount(self):
        res=self.con.execute('select sum(count) from cc').fetchone()
        if res==None: return 0
        return res[0]

    # the list of all categories
    def categories(self):
        cur=self.con.execute('select category from cc');
        return[d[0] for d in cur]

    # the train method takes an item and a classification.
    # it uses the getfeatures function of the class to break
    # the item into its separate features. it then calls
    # incf to increase the counts for this classification for every
    # feature. finally, it increases the total count for this classification.

    def train(self,item,cat):
        features=self.getfeatures(item)
        # increment hte count for every feature with this category
        for f in features:
            self.incf(f,cat)
            # increment the count for this category
        self.incc(cat)
        self.con.commit()

    # returns the probability that the specified item category
    # will contain that feature
    def fprob(self,f,cat):
        if self.catcount(cat)==0: return 0
        # the total number of times this feature appeared
        # in this category divided by the total number of items
        # in the category
        return self.fcount(f,cat)/self.catcount(cat)

    def weightedprob(self,f,cat,prf,weight=1.0,ap=0.5):
        # calculate the current probability
        basicprob=prf(f,cat)

        # count the number of times this feature has appeared in all cats
        totals=sum([self.fcount(f,c) for c in self.categories()])

        # calculate the weighted average
        bp = ((weight*ap)+(totals*basicprob))/(weight+totals)
        return bp

    def classify(self,item,default=None):
        probs={}
        # find the category with the highest probability
        max=0.0
        for cat in self.categories():
            probs[cat]=self.prob(item,cat)
            if probs[cat]>max:
                max=probs[cat]
                best=cat

        # ensure the probability exceeds the threshold*nextbest
        for cat in probs:
            if cat==best: continue
            if probs[cat]*self.getthreshold(best)>probs[best]: return default
        return best

class naivebayes(classifier):
    def docprob(self,item,cat):
        features=self.getfeatures(item)

        # multiply the probabilities of all the features together
        p=1
        for f in features: p*=self.weightedprob(f,cat,self.fprob)
        return p

    # the prob method calculates the probability of the category,
    # and returns the product of Pr(Document|Category) and Pr(Category)

    def prob(self,item,cat):
        catprob=self.catcount(cat)/self.totalcount()
        docprob=self.docprob(item,cat)
        return docprob*catprob
        
# this function will return the probability that an item with the specified
# featue belongs in the specified category, assuming there will be an equal
# number of items in each category


class fisherclassifier(classifier):
    def cprob(self,f,cat):
        # The frequency of this feature in this category    
        clf=self.fprob(f,cat)
        if clf==0: return 0

        # The frequency of this feature in all the categories
        freqsum=sum([self.fprob(f,c) for c in self.categories()])

        # The probability is the frequency in this category divided by
        # the overall frequency
        p=clf/(freqsum)
    
        return p
    
    def fisherprob(self,item,cat):
        # Multiply all the probabilities together
        p=1
        features=self.getfeatures(item)
        for f in features:
          p*=(self.weightedprob(f,cat,self.cprob))

        # Take the natural log and multiply by -2
        fscore=-2*math.log(p)

        return self.invchi2(fscore,len(features)*2)

    def invchi2(self,chi, df):
        m = chi / 2.0
        sum = term = math.exp(-m)
        for i in range(1, df//2):
            term *= m / i
            sum += term
        return min(sum, 1.0)

    def __init__(self,getfeatures):
        classifier.__init__(self,getfeatures)
        self.minimums={}

    def setminimum(self,cat,min):
        self.minimums[cat]=min

    def getminimum(self,cat):
        if cat not in self.minimums: return 0
        return self.minimums(cat)

    def classify(self,item,default=None):
        # loop through looking for the best result
        best=default
        max=0.0
        for c in self.categories():
          p=self.fisherprob(item,c)
          # make sure it exceeds its minimum
          if p>self.getminimum(c) and p>max:
              best=c
              max=p
        return best

    def invchi2(self,chi,df):
        m=chi/2.0
        sum=term=math.exp(-m)
        for i in range(1,df//2):
            term+=m/i
            sum+=term
        return min(sum,1.0)

    

    
    



    
    
            
