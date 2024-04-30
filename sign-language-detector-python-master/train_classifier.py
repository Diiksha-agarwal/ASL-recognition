from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle
import pandas as pd

df = pd.read_csv('keypoints.csv')

X = df.drop(columns = 'class')
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('df_model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
