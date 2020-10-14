import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

pos_5289 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/positive 5289/Kmer 2.npy"
neg_5289 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/negative 5289/Kmer 2.npy"
pos_1000 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/positive 1000/Kmer 2.npy"
neg_1000 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/negative 1000/Kmer 2.npy"

x_train_pos = np.load(pos_5289)
x_train_neg = np.load(neg_5289)
x_test_pos = np.load(pos_1000)
x_test_neg = np.load(neg_1000)

y_train_pos = np.tile(1, 5289)
y_train_neg = np.tile(0, 5289)
y_test_pos = np.tile(1, 1000)
y_test_neg = np.tile(0, 1000)

print(x_test_neg.shape)
print(y_test_neg.shape)

print(x_train_neg.shape)
print(y_train_neg.shape)

x_training = np.concatenate((x_train_pos, x_train_neg), axis=0)
y_training = np.concatenate((y_train_pos, y_train_neg))

x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
y_test = np.concatenate((y_test_pos, y_test_neg))

X_train, X_val, Y_train, Y_val = train_test_split(x_training, y_training, test_size=0.25, random_state=0)

seed = 10
np.random.seed(seed)

kf = KFold(n_splits=5, random_state=seed, shuffle=True)

test_avg_acc_list = []
j = 1
for c in range(1, 452, 50):
    RFC = RandomForestClassifier(n_estimators=c, max_depth=None)
    val_accuracy_list = []
    train_accuracy_list = []
    test_acc_list = []
    test_avg = 0
    for train_index, val_index in kf.split(x_training):
        X_train, X_val = x_training[train_index], x_training[val_index]
        y_train, y_val = y_training[train_index], y_training[val_index]

        RFC.fit(X_train, y_train)

        y_val_pred = RFC.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_accuracy_list.append(val_accuracy)

        y_train_pred = RFC.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracy_list.append(train_accuracy)

        y_test_pred = RFC.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_acc_list.append(test_accuracy)
        test_avg = test_avg + test_accuracy
    test_avg = test_avg / 5
    print(f"Epoch {j}/{10} for C = {c}")
    j = j + 1
    for i in range(0, len(train_accuracy_list)):
        print("Train Accuracy : ", train_accuracy_list[i], "\tValidation Accuracy : ", val_accuracy_list[i],
              "\tTest Accuracy ", test_acc_list[i])
    print("Avg Test Accuracy:", test_avg)

test_avg_acc_list = []
j = 1
for c in range(446, 456, 1):
    RFC = RandomForestClassifier(n_estimators=c, max_depth=None)
    val_accuracy_list = []
    train_accuracy_list = []
    test_acc_list = []
    test_avg = 0
    for train_index, val_index in kf.split(x_training):
        X_train, X_val = x_training[train_index], x_training[val_index]
        y_train, y_val = y_training[train_index], y_training[val_index]

        RFC.fit(X_train, y_train)

        y_val_pred = RFC.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_accuracy_list.append(val_accuracy)

        y_train_pred = RFC.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracy_list.append(train_accuracy)

        y_test_pred = RFC.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_acc_list.append(test_accuracy)
        test_avg = test_avg + test_accuracy
    test_avg = test_avg / 5
    print(f"Epoch {j}/{10} for C = {c}")
    j = j + 1
    for i in range(0, len(train_accuracy_list)):
        print("Train Accuracy : ", train_accuracy_list[i], "\tValidation Accuracy : ", val_accuracy_list[i],
              "\tTest Accuracy ", test_acc_list[i])
    print("Avg Test Accuracy:", test_avg)

test_avg_acc_list = []
j = 1
for c in range(1, 101, 10):
    RFC = RandomForestClassifier(n_estimators=500, max_depth=c)
    val_accuracy_list = []
    train_accuracy_list = []
    test_acc_list = []
    test_avg = 0
    for train_index, val_index in kf.split(x_training):
        X_train, X_val = x_training[train_index], x_training[val_index]
        y_train, y_val = y_training[train_index], y_training[val_index]

        RFC.fit(X_train, y_train)

        y_val_pred = RFC.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_accuracy_list.append(val_accuracy)

        y_train_pred = RFC.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracy_list.append(train_accuracy)

        y_test_pred = RFC.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_acc_list.append(test_accuracy)
        test_avg = test_avg + test_accuracy
    test_avg = test_avg / 5
    print(f"Epoch {j}/{10} for depth = {c}")
    j = j + 1
    for i in range(0, len(train_accuracy_list)):
        print("Train Accuracy : ", train_accuracy_list[i], "\tValidation Accuracy : ", val_accuracy_list[i],
              "\tTest Accuracy ", test_acc_list[i])
    print("Avg Test Accuracy:", test_avg)
