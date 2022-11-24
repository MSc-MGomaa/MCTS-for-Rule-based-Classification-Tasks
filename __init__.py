import produceClassifiers

if __name__ == '__main__':

    obj = produceClassifiers.Retrieve(dataset='Iris.csv', kFold=10, mValues=[.1, .5, 1, 5, 10, 20],
                                      jaccardValues=[.2, .5, .8], iterationsNumber=20,
                                      minimumSupport=10)
    obj.produce()

'''
The package call



def run(dataset, kFold, listOfmValues, listOfJaccardValues, iterationsNumber, minimumSupport):
    obj = produceClassifiers.Retrieve(dataset=dataset, kFold=kFold, mValues=listOfmValues,
                                      jaccardValues=listOfJaccardValues, iterationsNumber=iterationsNumber,
                                      minimumSupport=minimumSupport)
    obj.produce()

'''
