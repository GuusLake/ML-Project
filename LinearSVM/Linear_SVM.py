from sklearn.svm import LinearSVC
import numpy as np

def readData(fileName = 'mfeat-pix.txt'):
    """Reads data from a given filename and processes it 

    Args:
        fileName (str, optional): A string that specifies the path to a data file. Defaults to 'mfeat-pix.txt'.

    Returns:
        List: A list of tuples with two items: 
            -Item 0: A list of features (grascale values), 
            -Item 1: The number this list represents
    """    
    # Read data
    inputFile = open(fileName, 'r')
    dataDict = {}
    inputLines = inputFile.readlines()

    # Remove junk at front and start of line and split it into a list
    inputLines = [line.rstrip().lstrip().split('  ') for line in inputLines]

    # Turn list of Strings into a list of Ints
    for i in range(len(inputLines)):
        temp = []
        for j in inputLines[i]:
            temp.append(int(j))
        inputLines[i] = temp

    # Create list of tuples and determine number
    data = []
    for i in range(2000):
        data.append((inputLines[i], int(i/200)))
    return data

def splitData(data):
    """Splits the given data set into a train and test set (seperate features and their respective numbers)

    Args:
        data (List): A data set (in the format as is generated in readData())

    Returns:
        [List]: A list of numpy arrays with the train and test sets
    """
    # Create Empty lists  
    trainFeats = []
    trainNum = []
    testFeats = []
    testNum = []

    # Split data set in train and test set
    for i in range(2000):
        # 0-99 are train, 100-199 are test, etc
        if int(i / 100) % 2:
            testFeats.append(data[i][0])
            testNum.append(data[i][1])
        else:
            trainFeats.append(data[i][0])
            trainNum.append(data[i][1])
    return [np.array(trainFeats), np.array(trainNum), np.array(testFeats), np.array(testNum)]

def trainModel(data):
    """Trains a model and prints the accuracy

    Args:
        data (List): A data set (in the format as is generated in readData())
    """    
    trainFeats, trainNum, testFeats, testNum = splitData(data)
    model = LinearSVC(dual=False,  max_iter=100000, multi_class='crammer_singer')
    model.fit(trainFeats,trainNum)
    print(model.score(testFeats, testNum))

def main():
    data = readData()
    trainModel(data)

if __name__ == "__main__":
    main()