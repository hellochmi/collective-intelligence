# feedfilter.py
# parses rss feed

import feedparser
import re

# read loops over all the entries and uses the classifier to
# get a best guess at the classification, shows the best guess
# to the user then asks what the correct category should have been

# takes filename of url of a blog feed and classifies the entries
def read(feed,classifier):
    # get feed entries and loop over them
    f=feedparser.parse(feed)
    for entry in f['entries']:
        print
        print '-----'
        # print the contents of the entry
        print 'Title:     '+entry['title'].encode('utf-8')
        print
        print entry['description'].encode('utf-8')

        # combine all the text to create one item for the classifier
        fulltext='%s\n%s' % (entry['title'],entry['description'])

        # print the best guess at the current category
        print 'Guess: '+str(classifier.classify(entry))

        # ask the user to specify the correct category and train on that
        cl=raw_input('Enter category: ')
        classifier.train(entry,cl)

# entryfeatures extracts the words from the title and the summary, just like
# getwords did earlier. it marks all the words int he title as such and adds them
# as features. the words int eh summary are added as features, and then pairs of
# consecutive words are added as well. the function adds the creator and
# publisher as features without dividing them up, and finally, it counts the
# number of words in the summary that are uppercase. if more than 30 percent of
# the words are uppercase, the function adds an additional feature called
# uppercase to the set

def entryfeatures(entry):
    splitter=re.compile('\\W*')
    f={}

    # extract the title words and annotate
    titlewords=[s.lower() for s in splitter.split(entry['title'])
                if len(s)>2 and len(s)<20]
    for w in titlewords: f['Title:'+w]=1

    # extract the summary words
    descriptionwords=[s.lower() for s in splitter.split(entry['description'])
                if len(s)>2 and len(s)<20]

    # count uppercase words
    uc=0
    for i in range(len(descriptionwords)):
        w=descriptionwords[i]
        f[w]=1
        if w.isupper(): uc+=1

        # get word pairs in summary as features
        if i<len(descriptionwords)-1:
            twowords=' '.join(descriptionwords[i:i+1])
            f[twowords]=1

    # uppercase virtual word flagging
    if float(uc)/len(descriptionwords)>0.3: f['UPPERCASE']=1

    return f
