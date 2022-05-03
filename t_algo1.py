import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")

time_list = []
play_list = []
k = 0;

for i,j in zip(df['open'], df['close']):
    if i < j:
        item = 'H'
        play_list.append(item)
    else:
        item = 'L'
        play_list.append(item)

    time_list.append(int(df.minute.str.split().tolist()[k][1].replace(':','')))
    k += 1;

df['minute'] = time_list
df.drop(df.columns[[0]], axis=1, inplace=True)

df.insert(loc=len(df.columns),column='play',value=play_list)

inputs = df.drop('play',axis='columns')
target = df.play

X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)

model.predict([[1026]])
