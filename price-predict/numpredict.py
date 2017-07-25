from random import random, randint
import math

def wineprice(rating,age):
    peak_age=rating-50

    # calculate price based on rating
    price=rating/2
    if age>peak_age:
        # past its peak, goes bad in 5 years
        price=price*(5-(age-peak_age))
    else:
        # increases to 5x original value as it approaches its peak
        price=price*(5*((age+1)/peak_age))
    if price<0: price=0
    return price

def wineset1():
    rows=[]
    for i in range(300):
        # create random age and rating
        rating=random()*50+50
        age=random()*50

        # get reference price
        price=wineprice(rating,age)

        # add some noise
        price*=(random()*0.4+0.8)

        # add to the dataset
        rows.append({'input':(rating,age),
                     'result':price})
    return rows

# measure similarity using euclidean distance

def euclidean(v1,v2):
    d=0.0
    for i in range(len(v1)):
        d+=(v1[i]-v2[i])**2
    return math.sqrt(d)

# get the distances between a given item and every item in the original dataset

def getdistances(data,vec1):
    distancelist=[]
    for i in range(len(data)):
        vec2=data[i]['input']
        distancelist.append((euclidean(vec1,vec2),i))
    distancelist.sort()
    return distancelist

def knnestimate(data,vec1,k=3):
    # get sorted distances
    dlist=getdistances(data,vec1)
    avg=0.0

    # take the average of the top k results
    for i in range(k):
        idx=dlist[i][1]
        avg+=data[idx]['result']
    avg=avg/k
    return avg

def inverseweight(dist,num=1.0,const=1.0):
    return num/(dist+const)

def subtractweight(dist,const=1.0):
    if dist>const:
        return 0
    else:
        return const-dist

def gaussian(dist,sigma=10.0):
    return math.e**(-dist**2/(2*sigma**2))

# calculates the weighted average by multiplying each item
# by its weight before adding them together

def weightedknn(data,vec1,k=5,weightf=gaussian):
    # get distances
    dlist=getdistances(data,vec1)
    avg=0.0
    totalweight=0.0

    # get weighted average
    for i in range(k):
        dist=dlist[i][0]
        idx=dlist[i][1]
        weight=weightf(dist)
        avg+=weight*data[idx]['result']
        totalweight+=weight
    avg=avg/totalweight
    return avg

def dividedata(data,test=0.05):
    trainset=[]
    testset=[]
    for row in data:
        if random()<test:
            testset.append(row)
        else:
            trainset.append(row)
    return trainset,testset

def testalgorithm(algf,trainset,testset):
    error=0.0
    for row in testset:
        guess=algf(trainset,row['input'])
        error+=(row['result']-guess)**2
    return error/len(testset)

def crossvalidate(algf,data,trials=100,test=0.05):
    error=0.0
    for i in range(trials):
        trainset,testset=dividedata(data,test)
        error+=testalgorithm(algf,trainset,testset)
    return error/trials

def wineset2():
    rows=[]
    for i in range(300):
        rating=random()*50+50
        age=random()*50
        aisle=float(randint(1,20))
        bottlesize=[375.0,750.0,1500.0,3000.0][randint(0,3)]
        price=wineprice(rating,age)
        price*=(bottlesize/750)
        price*=(random()*0.9+0.2)
        rows.append({'input':(rating,age,aisle,bottlesize),'result':price})
    return rows

def rescale(data,scale):
    scaleddata=[]
    for row in data:
        scaled=[scale[i]*row['input'][i] for i in range(len(scale))]
        scaleddata.append({'input':scaled,'result':row['result']})
    return scaleddata

def createcostfunction(algf,data):
    def costf(scale):
        sdata=rescale(data,scale)
        return crossvalidate(algf,sdata,trials=10)
    return costf

weightdomain=[(0,20)]*4

def wineset3():
    rows=wineset1()
    for row in rows:
        if random()<0.5:
            # wine was bought at a discount store
            row['result']*=0.6
    return rows

def probguess(data,vec1,low,high,k=5,weightf=gaussian):
    dlist=getdistances(data,vec1)
    nweight=0.0
    tweight=0.0

    for i in range(k):
        dist=dlist[i][0]
        idx=dlist[i][1]
        weight=weightf(dist)
        v=data[idx]['result']

        if v>=low and v<=high:
            nweight+=weight
        tweight+=weight
    if tweight==0: return 0

    return nweight/tweight

def cumulativegraph(data,vec1,high,k=5,weightf=gaussian):
    t1=arange(0.0,high,0.1)
    cprob=array([probguess(data,vec1,0,v,k,weightf) for v in t1])
    plot(t1,cprob)
    show()






