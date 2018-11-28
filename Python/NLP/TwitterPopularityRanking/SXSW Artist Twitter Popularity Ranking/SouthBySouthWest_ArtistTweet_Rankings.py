#Jesse Broussard - CSCE561 - Twitter Event Text Parsed Entity Popularity 
#Implementation of Artist Popularity Determination using SXSW Twitter feed 2017, and 10000 (really around 8000) MTV Music Artists list

#In this scenario I'm treating artists as the dictionary of words which allows me
#to treat each artist as a word in a traditional NLP sense.  
import csv;
import numpy as np;
from datetime import datetime

#Twitter User Data:
#Tweets may be by non unique users, so we must index all users
#then using the twitter handle, we can find that index again for updating
#values in our datastructures correctly.
users = [];
userIndexFinder = {};
uCount = 0;

#SXSW Artist Data:
#all artists of SXSW listed on their website for 2017.
SXSW_artists = [];
SXSW_artistsIndexFinder = {};
sCount = 0;

RSort = []

def rSortSet(Score,bandName):
    global RSort

    RSort[0].append([Score,bandName]);

def rSortSort():
    global RSort
    
    RSort[0].sort(key=lambda x: x[0],reverse=True)

def SXSW_artistsCollect():
    global SXSW_artists
    global SXSW_artistsIndexFinder
    global sCount
    global RSort
    
    lineCount = 0
    with open('SXSW_HipHop.csv', 'r',encoding = 'utf8') as csvFile:
        lineReader = csv.reader(csvFile, delimiter=',')
        for line in lineReader:
            temp = line[0]
            SXSW_artists.append(temp.lower())
            SXSW_artistsIndexFinder[temp.lower()] = sCount;
            sCount += 1

    for i in range(1):
        RSort.append([])



def gatherTweetStatistics():
    global users
    global userIndexFinder 
    global uCount
    

    lineCount = 0
    with open('SXSW.csv', 'r', errors='ignore') as csvFile:
        lineReader = csv.reader(csvFile, delimiter=',')
        for line in lineReader:
            if lineCount != 1:
                currentUser = line[0][1:len(line[0])].lower();
                #Has this user tweeted before?
                if(currentUser not in userIndexFinder):
                    #initiate new user
                    users.append(currentUser.lower());
                    userIndexFinder[currentUser.lower()] =  uCount;
                    uCount += 1;

                

        lineCount +=1 ;





def main():
    
    global users
    global userIndexFinder
    global uCount
    global SXSW_artists
    global SXSW_artistsIndexFinder
    global sCount
    global RSort
    
    gatherTweetStatistics();
    SXSW_artistsCollect();

    tweetMatrix = np.empty(shape=(sCount,uCount),dtype=np.float32);

    lineCount = 0


    print("Making Initial Scores...")
    #initial scoring  
    with open('SXSW.csv','r', encoding= "UTF-8",errors="ignore") as csvFile:
        lineReader = csv.reader(csvFile,delimiter=",")
        lineCount = 0;

        start=datetime.now()

        print("Progress:",end='')
        #for entry in SXSW.csv
        for line in lineReader:
            #Get tweeting user's handle
            currentUser = line[0][1:len(line[0])].lower()
            if lineCount!=0:

                #split the tweet, and remove duplicate words (unique tokens only)
                tweet = set(line[1].lower().split())

                
                for tweetWord in tweet:
                    for artist in SXSW_artists:
                        leng = len(artist.split(" "))
                        lCount = 0
                        tempScore = 0;
                        
                        for token in artist.split(" "):
                            if token in tweet:
                                lCount+=1
                                
                        if(lCount == leng):
                            tempScore += (eval(line[2]) + eval(line[4]))
                            tweetMatrix[SXSW_artistsIndexFinder[artist],userIndexFinder[currentUser]] += tempScore
                            
                                
                    
                '''
                for temp2 in SXSW_artists:
                    for temp3 in temp2.split(" "): 
                        
                        leng = len(temp2.split(" "))
                        lCount = 0
                        tempScore = 0;

                        #Progress bar
                        
                            
                        for temp4 in temp1:
                            if(temp3 in temp4.lower()): 
                          
                                lCount += 1
                                #add (followers + retweets)/# of pieces in band name
                                tempScore += (eval(line[2]) + eval(line[4])/leng);
                                #print("Temp Score:" , tempScore)
                                
                        if (lCount == leng):
                            tweetMatrix[SXSW_artistsIndexFinder[temp2],userIndexFinder[currentUser]] += tempScore
                            #print("Tweet:", temp1)
                            #print("Kept Score Added:", tempScore, " New Total for ", temp2, ":",tweetMatrix[SXSW_artistsIndexFinder[temp2],userIndexFinder[currentUser]])
                    '''
            #Progress bar
            if lineCount%1000 == 0:
                print("=",end='')
            lineCount += 1
        print("|")
        print("Time Elapsed: ", datetime.now()-start)

    
    print("Normalizing Tweet Mentions By Tweet Count Per User:")

    start=datetime.now()

    tCounts = np.empty(shape=(uCount))
    for t in range(uCount):
        tempCount = 0;
        for y in range (sCount):
            if (tweetMatrix[y,t] > 0):
                tempCount += 1
        for y in range (sCount):
            if (tweetMatrix[y,t] > 0):
                tweetMatrix[y,t] = (tweetMatrix[y,t] / tempCount)
        
    
    scores = np.sum(tweetMatrix, axis=1);
    
    print("Time Elapsed: ", datetime.now()-start)





    
    c = 0;
    for x in scores:
        if x != 0:
            rSortSet(x,SXSW_artists[c])
        c+=1
        
    rSortSort()

    for p in range(15):
        print("- Artist ", format(RSort[0][p][1], '20s'), " - Rank " ,format(p+1, '3.0f'), " - Score ",format(RSort[0][p][0],'12.2f'), " -")
        
    #print(SXSW_artists[c],":",x)
    

main()

