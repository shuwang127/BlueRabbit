import pandas as pd
import numpy as np
import math
import string

# global path.
rootPath = './'
dataPath = rootPath + '/data/'
tempPath = rootPath + '/temp/'

def main():
    print('=================================== BlueRabbit Team ===================================')
    print('[Info] Process the news documents ...')
    locKeywords, raceKeywords, contents = ReadData()
    print('[Info] Information from the title and contents ...')
    print(contents)
    print('[Info] Information from the county and state ...')
    print(locKeywords)
    print('[Info] Information from the race ...')
    print(raceKeywords)
    print('---------------------------------------------------------------------------------------')
    dsetNew, labels, county = Process()
    prior, likelihood = TrainNaiveBayes(dsetNew, labels)
    print('[Para] Model parameters:')
    print(prior)
    print(likelihood)
    probs = TestNaiveBayes(prior, likelihood, dsetNew)
    print('[Info] Get the first-order probabilities from evaluation.')
    print('---------------------------------------------------------------------------------------')
    print('[Info] Combining the information from news.')
    probs = ProbsAdjust(probs, county, dsetNew, locKeywords, raceKeywords, contents)
    print('[Info] Get the second-oder probabilities from evaluation.')
    Write2File(probs)
    print('[Info] Analysis done!')
    print('---------------------------------------------------------------------------------------')
    filename = dataPath + 'Roe-Sepowitzd et al (2019).txt'
    print('[Info] Input a new text file: ' + filename + '.')
    keywords = ReadFile(filename)
    print('[Info] Get the keywords (w. weights) from the text:')
    print(keywords)
    print('=================================== BlueRabbit Team ===================================')
    return

# train the naive bayes model.
def TrainNaiveBayes(features, labels):
    '''
    train the naive bayes model.
    :param features: training set features
    :return: model parameters - prior, likelihood
    '''
    # define the log prior.
    def GetLogPrior(labelTrain):
        # count the number.
        nTotal = len(labelTrain)
        nPos = list(labelTrain).count(1)
        nNag = list(labelTrain).count(0)
        # calculate the logprior.
        priorPos = math.log(nPos / nTotal)
        priorNag = math.log(nNag / nTotal)
        prior = [priorNag, priorPos]
        return prior

    # define loglikelihood.
    def GetLogLikelihood(features, labelTrain):
        # get V and D.
        V = len(features[0])
        D = len(features)
        cls = 2
        # initilaze likelihood matrix.
        likelihood = np.zeros((cls, V))
        for ind in range(D):
            for i in range(V):
                likelihood[labelTrain[ind]][i] += features[ind][i]
        # Laplace smoothing.
        denom = np.zeros((cls, 1))
        for lb in range(cls):
            denom[lb] = sum(likelihood[lb]) + V
            for i in range(V):
                likelihood[lb][i] += 1
                likelihood[lb][i] /= denom[lb]
                likelihood[lb][i] = math.log(likelihood[lb][i])
        return likelihood

    # get the log prior.
    prior = GetLogPrior(labels)
    # get the log likelihood
    likelihood = GetLogLikelihood(features, labels)
    return prior, likelihood

# test and evaluate the performance.
def TestNaiveBayes(prior, likelihood, featTest):
    # get predictions for testing samples with model parameters.
    def GetPredictions(prior, likelihood, featTest):
        # get V and D.
        V = len(featTest[0])
        D = len(featTest)
        cls = 2
        T = 0.015
        # get pred(D, cls) matrix and predictions(D, 1).
        pred = np.zeros((D, cls))
        predictions = np.zeros((D, 1))
        probs = np.zeros((D, 1))
        for ind in range(D):
            for lb in range(cls):
                pred[ind][lb] += prior[lb]
                for i in range(V):
                    pred[ind][lb] += likelihood[lb][i] * featTest[ind][i]
            predictions[ind] = list(pred[ind]).index(max(pred[ind]))
            p0 = pred[ind][0] / (pred[ind][0] + pred[ind][1])
            p1 = pred[ind][1] / (pred[ind][0] + pred[ind][1])
            probs[ind] = math.exp(p1/T) / (math.exp(p0/T) + math.exp(p1/T))
        return predictions, probs

    # get predictions for testing samples.
    predictions, probs = GetPredictions(prior, likelihood, featTest)
    return probs

def ProbsAdjust(probs, county, dset, locKeywords, raceKeywords, contents):
    # adjust information from locKeywords
    for i in range(len(probs)):
        # locKeywords.
        p = 0
        if (county[i] in locKeywords):
            p = locKeywords[county[i]] / (sum(locKeywords.values()))
        t = (math.exp(p) - 1) / (math.exp(1) - 1)
        probs[i] = (1 - t) * probs[i] + t
    # adjust information from contents.
    for i in range(len(probs)):
        # contents
        p = 0
        if (county[i] in contents):
            p = contents[county[i]] / (sum(contents.values()))
        t = (math.exp(p) - 1) / (math.exp(1) - 1)
        probs[i] = (1 - t) * probs[i] + t
    # race
    for i in range(len(probs)):
        p = raceKeywords['White'] / (sum(raceKeywords.values())) * dset[i][-4] / 100
        t = (math.exp(p) - 1) / (math.exp(1) - 1)
        probs[i] = (1 - t) * probs[i] + t
    return probs

def ReadData():
    # read csv file.
    filePath = dataPath + '/NewsData.csv'
    dset = pd.read_csv(filePath)
    # weights
    numCase = [int(item) if is_number(item) else 1 for item in dset['case']]
    numCriminals = [int(item) if is_number(item) else 1 for item in dset['#Criminals']]
    numClass = [2 if item == 'havecase' else 1 for item in dset['Category']]
    weights = [numCase[i] * numCriminals[i] * numClass[i] for i in range(len(numCase))]
    # get locations.
    locKeywords = dict()
    for i in range(len(weights)):
        loc = dset['traffic/aresst_location'][i]
        if loc not in locKeywords.keys():
            locKeywords[loc] = 0
        locKeywords[loc] += weights[i]
    locKeywords = SortDict(locKeywords)
    # get race.
    raceKeywords = dict()
    for i in range(len(weights)):
        race = dset['c_race'][i]
        if race not in raceKeywords.keys():
            raceKeywords[race] = 0
        raceKeywords[race] += weights[i]
    raceKeywords = SortDict(raceKeywords)
    # get content and title.
    counties = GetCounty()
    contents = dict()
    for i in range(len(weights)):
        text = dset['title'][i] + ' ' + dset['Content'][i]
        text = string.capwords(text)
        for county in counties:
            num = text.count(county)
            if num != 0:
                if county not in contents.keys():
                    contents[county] = 0
                contents[county] += num * weights[i]
    contents = SortDict(contents)
    # return
    return locKeywords, raceKeywords, contents

def ReadFile(filename):
    locKeywords, raceKeywords, contents = ReadData()
    # get keyword list
    keywordList = []
    keywordList.extend([item for item in locKeywords if locKeywords[item] >= 8])
    keywordList.extend([item for item in raceKeywords if raceKeywords[item] >= 10])
    keywordList.extend([item for item in contents if contents[item] >= 30])
    keywordListExt = ['customer', 'white', 'male', 'business model', 'post', 'online', 'advertise', 'transport to',\
                      'hotel', 'motel', 'apartment, book', 'telephone', 'phone number', 'eastern district of Virginia', \
                      'Maryland', 'Washington DC', 'strip club', 'bar', 'massage parlor', 'hotspot', 'commercial sex act',\
                      'sex', 'trafficking', 'high income', 'earnings']
    keywordList.extend([string.capwords(item) for item in keywordListExt])
    keywordList = list(set(keywordList))
    # read file.
    fp = open(filename, encoding='utf-8')
    text = fp.read()
    text = string.capwords(text)
    # statistic.
    keywords = dict()
    for item in keywordList:
        if (type(item) == str):
            num = text.count(item)
            if num != 0:
                keywords[item] = num
    keywords = SortDict(keywords)
    # return
    return keywords

def Process():
    filename = dataPath + '/datahub.csv'
    dset = pd.read_csv(filename)
    dsetNew = pd.DataFrame()
    # price_room_per_night
    dsetNew['price_room_per_night'] = [int(item[1:]) for item in dset['price_room_per_night']]
    # star
    dsetNew['Star'] = dset['Star']
    # city_has_criminal_recorded_courtcase
    dsetNew['city_has_criminal_recorded_courtcase'] = dset['city_has_criminal_recorded_courtcase']
    # hotspot_Paloris
    dsetNew['hotspot_Paloris'] = dset['hotspot_Paloris']
    # strip_club_40min
    dsetNew['strip_club_40min'] = dset['strip_club_40min']
    # # massage_parlor_nearby
    dsetNew['# massage_parlor_nearby'] = dset['# massage_parlor_nearby']
    # identified_online_ads
    dsetNew['identified_online_ads'] = dset['identified_online_ads']
    # recent_has_sport_event
    dsetNew['recent_has_sport_event'] = dset['recent_has_sport_event']
    # race_white_more_than_60%
    dsetNew['race_white_more_than_60%'] = dset['race_white_more_than_60%']
    # male_avg_Earning
    dsetNew['male_avg_Earning'] = [int(item[1:-4]) * 1000 + int(item[-3:]) for item in dset['male_avg_Earning']]
    # Single_Rate
    dsetNew['Single_Rate'] = [float(item[0:-1]) for item in dset['Single_Rate']]
    # white_male
    dsetNew['white_male'] = dset['white_male']
    # white_average
    dsetNew['white_average'] = [float(item[0:-1]) for item in dset['white_average']]
    # male_rate
    dsetNew['male_rate'] = [float(item[0:-1]) for item in dset['male_rate']]
    # marriage_rate
    dsetNew['marriage_rate'] = [float(item[0:-1]) for item in dset['marriage_rate']]
    # Ave_Earning
    dsetNew['Ave_Earning'] = [int(item[1:-4]) * 1000 + int(item[-3:]) for item in dset['Ave_Earning']]
    dsetNew = dsetNew.values.tolist()
    # list.
    filename = dataPath + '/labels.csv'
    labels = pd.read_csv(filename)
    labels = labels.values.tolist()
    labels = [item[0] for item in labels]
    return dsetNew, labels, dset['County_seat'].values.tolist()

def GetCounty():
    # get counties.
    df = pd.read_csv(dataPath + '/laucnty16.csv')
    counties = [item.split(',')[0] for item in df['County Name/State Abbreviation'] if ', VA' in item]
    counties = [item[:-7] if ' County' in item else item for item in counties]
    counties = [item[:-5] if ' city' in item else item for item in counties]
    # add.
    countiesExtend = ['Springfield', 'Sterling', 'Alexandria', 'Woodbridge', 'Centreville', 'Chantilly'\
                      'Fairfax', 'Herdon', 'Tyson', 'Ashburn', 'Manassas', 'Vienna']
    counties.extend(countiesExtend)
    # delete.
    counties.remove('Washington')
    # unorder.
    counties = list(set(counties))
    # print(counties)
    return counties

def is_number(s):
    try:
        float(s)
        if math.isnan(float(s)):
            return False
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def SortDict(d):
    dNew = dict()
    while (len(d)):
        item = max(d, key=d.get)
        dNew[item] = d[item]
        d.pop(item)
    return dNew

def Write2File(probs):
    # read csv file.
    filePath = dataPath + '/datahub.csv'
    dset = pd.read_csv(filePath)
    dset['risk'] = probs
    dset.to_csv(dataPath + '/risk.csv')
    print('[Info] Results have been saved in ' + dataPath + '/risk.csv')
    return

if __name__ == '__main__':
    main()