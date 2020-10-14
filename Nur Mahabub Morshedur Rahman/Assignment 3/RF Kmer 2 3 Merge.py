import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

a_pos_5289 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/positive 5289/Kmer 2.npy"
a_neg_5289 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/negative 5289/Kmer 2.npy"
a_pos_1000 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/positive 1000/Kmer 2.npy"
a_neg_1000 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/negative 1000/Kmer 2.npy"

a_x_train_pos = np.load(a_pos_5289)
a_x_train_neg = np.load(a_neg_5289)
a_x_test_pos = np.load(a_pos_1000)
a_x_test_neg = np.load(a_neg_1000)

b_pos_5289 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/positive 5289/Kmer 3.npy"
b_neg_5289 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/negative 5289/Kmer 3.npy"
b_pos_1000 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/positive 1000/Kmer 3.npy"
b_neg_1000 = "C:/Users/Nur Rafi/Documents/Data Sets/Pattern Laboratory/Assignment 2/negative 1000/Kmer 3.npy"

b_x_train_pos = np.load(b_pos_5289)
b_x_train_neg = np.load(b_neg_5289)
b_x_test_pos = np.load(b_pos_1000)
b_x_test_neg = np.load(b_neg_1000)

y_train_pos = np.tile(1, 5289)
y_train_neg = np.tile(0, 5289)
y_test_pos = np.tile(1, 1000)
y_test_neg = np.tile(0, 1000)

print(a_x_test_neg.shape)
print(a_x_train_neg.shape)
print(a_x_test_pos.shape)
print(a_x_train_pos.shape)
print(b_x_test_neg.shape)
print(b_x_train_neg.shape)
print(b_x_test_pos.shape)
print(b_x_train_pos.shape)

a_x_training = np.concatenate((a_x_train_pos, a_x_train_neg), axis=0)
a_x_test = np.concatenate((a_x_test_pos, a_x_test_neg), axis=0)

b_x_training = np.concatenate((b_x_train_pos, b_x_train_neg), axis=0)
b_x_test = np.concatenate((b_x_test_pos, b_x_test_neg), axis=0)

y_training = np.concatenate((y_train_pos, y_train_neg))
y_test = np.concatenate((y_test_pos, y_test_neg))

x_training = np.concatenate((a_x_training, b_x_training), axis=1)

x_test = np.concatenate((a_x_test, b_x_test), axis=1)

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
for c in range(346, 356, 1):
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
    RFC = RandomForestClassifier(n_estimators=270, max_depth=c)
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
