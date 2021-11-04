import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import os
from azureml.core import Run
 
parser = argparse.ArgumentParser("train")
 
parser.add_argument("--train", type=str, help="train")
parser.add_argument("--test", type=str, help="test")
parser.add_argument("--model", type=str, help="model")
 
args = parser.parse_args()
run = Run.get_context()
 
run.log("Training start time", str(datetime.datetime.now()))

train=np.loadtxt(args.train+"/train.txt",dtype=float)
test=np.loadtxt(args.test+"/test.txt",dtype=float)
 
X_train=train[:,:-1]
Y_train=train[:,-1]
 
X_test=test[:,:-1]
Y_test=test[:,-1]
 
param_map = {"max_depth":[2,5,10,25], 
             "min_samples_leaf":[2,5,10,50]}
model= RandomForestClassifier()
best= GridSearchCV(model, param_map, verbose = 1, n_jobs = 1).fit(X_train, Y_train)

result = best.score(X_test, Y_test)

Y_pred = best.predict(X_test)
result1 = classification_report(Y_test, Y_pred, target_names=['0','1'], output_dict=True)
precision = result1['0']['precision']
recall = result1['0']['recall']
accuracy = result1['accuracy']

run.log('Score :', result)
run.log('Precision:', precision)
run.log('Recall:', recall)
run.log('Accuracy:', accuracy)
run.complete()