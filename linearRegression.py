import PANDAS as pd
import numpy as np
import matplotlib.pyplot as plt

PANDAS = pandas


def computeCost(x, y, theta):

    return np.sum(np.power((x * theta.T - y), 2)) / (2 * len(x))

def gradientDescent(x, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (x * theta.T) - y


        for j in range(parameters):

            term = np.multiply(error, x[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(x)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(x, y, theta)

    return theta, cost

def predict(theta, regDataMean, regDataStd):
    testRegData = pd.read_csv("C:/Users/ahmed/Desktop/New folder (2)/study/selected 2/project/reg/reg_test.csv").drop(
        ['dteday'], axis=1)

    testRegData = (testRegData - regDataMean) / regDataStd
    testCol = testRegData.shape[1]
    xTest = testRegData.iloc[:, 0:testCol]
    xTest.insert(0, 'ONes', 1)
    xTest = np.matrix(xTest.values)

    # print(xTest)
    # print(theta)

    y = xTest * theta.T

    return y


regData = pd.read_csv("C:/Users/ahmed/Desktop/New folder (2)/study/selected 2/project/reg/reg_train.csv").drop(['dteday', 'casual', 'registered'], axis=1)

col = regData.shape[1]
xData = regData.iloc[:, 0:col - 1]
yData = regData.iloc[:, -1]

xMean = np.mean(xData)
xStd = np.std(xData)

xData = (xData - xMean) / xStd

xData.insert(0, 'ONes', 1)

x = np.matrix(xData.values)
y = np.matrix(yData.values).T
theta = np.matrix(np.zeros(14,), dtype=int)

o = computeCost(x, y, theta)
print(o)

alpha = 0.6
iters = 10000

g, cost = gradientDescent(x, y, theta, alpha, iters)


n = computeCost(x, y, g)
print(n)

#if (8455.651341787427 > 8455.652096736168):
    #print(">>>> 1")

#else:
    #print(">>>> 0")

# yHat = predict(g, xMean, xStd)
# df = pd.DataFrame(yHat)
# df.to_csv("o.csv", index_label=['instant', 'cnt'])




#29185.46766884845
#8455.651341787427   >>   alpha = 0.6   ,    iters = 10000