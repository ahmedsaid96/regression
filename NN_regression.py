import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

trainData = pd.read_csv("C:/Users/ahmed/Desktop/New folder (2)/study/selected 2/project/reg/reg_train.csv").drop(['dteday', 'casual', 'registered'], axis=1)
testData = pd.read_csv("C:/Users/ahmed/Desktop/New folder (2)/study/selected 2/project/reg/reg_test.csv").drop(['dteday'], axis=1)

xTrain = trainData.drop('cnt', axis=1)
y = trainData['cnt']
X_train, X_test, y_train, y_test = train_test_split(xTrain, y, test_size=0.33)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

reg = MLPRegressor(max_iter=10000, learning_rate_init=0.01)
reg.fit(X_train, y_train)
out = reg.predict(X_test)
print(reg.score(X_test, y_test))
# ins = testData['instant']
# d = {'instant': ins, 'cnt': out}
# df = pd.DataFrame(d)
#
# df.to_csv("NNout.csv")


# print(xTrain)
# print('______________________________________________________________________')
# print(xTest)
# print('______________________________________________________________________')
# print(x)

