import pandas as pd
import numpy as np
import math as math
def train_data(train,test):
    list_check = []
    for i in range(len(train)):
        if i == len(train)-1:
            list_prediction = test[i]
            #print(test[i])
        else:
            list_check.append(test[i])
            #print(test[i])
    length = len(test)
    print(list_check)
    # train = all data
    # test = all test
    # list check = all test but decision column
    # list predection last column of test, the column to compare to
    ##hamming_distance(train,test,list_check,list_prediction)
    ##naive_bayes(train,test,list_check,list_prediction)
    id3(train, test, list_check, list_prediction,defult)

def hamming_distance(train,test,list_check,list_prediction):
    idx = 0
    list_knn_result = []
    #print(range(len(list_check[0])))
    print(list_knn_result)
    for k in range(len(list_check[idx])):
        class_type = list_check[idx][k]
        age_type = list_check[idx + 1][k]
        gender_type = list_check[idx + 2][k]
        #enter tyuv

        idx =0
        list = []
        counter = 0
        u=0
        for j in range(len(train[u])):
            t_class_type = train[u][j]
            t_age_type = train[u+1][j]
            t_gender_type = train[u+2][j]
            t_decision = train[u + 3][j]
            if class_type !=t_class_type:
                counter +=1
            if age_type != t_age_type:
                counter +=1
            if gender_type !=t_gender_type:
                counter+=1
            if counter == 0:
                list.append((counter,t_decision))
            else:
                list.append((counter,t_decision))
            counter = 0
        #list.sort()
        import operator
        list.sort(key=operator.itemgetter(0))

        yes = 0
        no = 0
        new_list = [x[1] for x in list]

        for k in range(5):
            if new_list[k] == 'no':
                no += 1
            else:
                yes +=1
        if yes > no:
            list_knn_result.append('yes')
        else:
            list_knn_result.append('no')
    calcAccuracy(test, list_knn_result)

def naive_bayes(train,test,list_check,list_prediction):
    list_naive_bayes_result = []
    yes = no = p_train_yes = p_train_no = idx = 0
    for i in range(len(train)):
        if i == len(train) - 1:
            list_prediction = train[i]
    for j in list_prediction:
        if j == 'yes':
            yes += 1
        else:
            no += 1
    p_train_yes = float(yes/len(train[0]))
    p_train_no = float(no/len(train[0]))
    for k in range(len(list_check[idx])):
        class_type = list_check[idx][k]
        age_type = list_check[idx + 1][k]
        gender_type = list_check[idx + 2][k]

        idx = 0
        list = []
        counterYes1 = counterYes2 = counterYes3 = 0
        counterNo1 = counterNo2 = counterNo3 = 0
        prob1_yes = prob2_yes = prob3_yes = 0
        prob1_no = prob2_no = prob3_no = 0
        p_example_yes = p_example_no = 0
        u = 0
        train_len = len(train[u])
        for j in range(len(train[u])):
            t_class_type = train[u][j]
            t_age_type = train[u + 1][j]
            t_gender_type = train[u + 2][j]
            classification = train[u + 3][j]
            if class_type == t_class_type and classification == 'yes':
                counterYes1 += 1
            if age_type == t_age_type and classification == 'yes':
                counterYes2 += 1
            if gender_type == t_gender_type and classification == 'yes':
                counterYes3 += 1
            if class_type == t_class_type and classification == 'no':
                counterNo1 += 1
            if age_type == t_age_type and classification == 'no':
                counterNo2 += 1
            if gender_type == t_gender_type and classification == 'no':
                counterNo3 += 1

        prob1_yes = float(counterYes1/yes)
        prob2_yes = float(counterYes2 / yes)
        prob3_yes = float(counterYes3 / yes)
        prob1_no = float(counterNo1 / no)
        prob2_no = float(counterNo2 / no)
        prob3_no = float(counterNo3 / no)

        p_example_yes = p_train_yes * prob1_yes * prob2_yes * prob3_yes
        p_example_no = p_train_no * prob1_no * prob2_no * prob3_no

        if p_example_no > p_example_yes:
            list_naive_bayes_result.append('no')
        else:
            list_naive_bayes_result.append('yes')
    print(list_naive_bayes_result)
    calcAccuracy(test,list_naive_bayes_result)

def id3(train,test,list_check,list_prediction,defult):
    list_id3 = []
    train_without_decision = []
    yes = no = p_train_yes = p_train_no =  0
    for i in range(len(train)):
        if i == len(train) - 1:
            # classify of last column in train
            list_prediction = train[i]
        else:
            train_without_decision.append(train[i])

    for j in list_prediction:
        if j == 'yes':
            yes += 1
        else:
            no += 1
    if yes > no:
        defult = 'yes'
    else:
        defult = 'no'
    len_train = len(train[0])
    decision_entropy = calc_entropy(yes,no,len_train)
    attributes_gain = []
    for i in train_without_decision:
        sub_entropy = check_attribute(i,list_prediction)
        attributes_gain.append(decision_entropy-sub_entropy)
    print(attributes_gain)
    # הגעתי למצב שיש לי את הגיין מהאטרביוטים ואני צריכה לבחור עם מי ללכת שיהיו בראש העץ \
    #ועכשיו אני צריכה לבחור מי הבא שיהיה וזאת הרקורסיה כנראה
    # p_train_yes = float(yes / len(train[0]))
    # p_train_no = float(no / len(train[0]))

def check_attribute(column_attribute,list_prediction):
    values = set(column_attribute)
    positive = negative = 0
    entropies = []
    for value in values:
        len_examples = len(column_attribute)
        for val in range(len(column_attribute)):
            if column_attribute[val] == value and list_prediction[val] == 'yes':
                positive += 1
            elif column_attribute[val] == value and list_prediction[val] == 'no':
                negative += 1
            #all examples of this values
        total = negative+positive
        fraq = float(total/len_examples)
        answer = fraq*calc_entropy(positive,negative,total)
        entropies.append(answer)
        positive = negative = total = 0
    final = 0
    for entropy in entropies:
             final = final + entropy
    print(final)
    return final


def calc_entropy(yes,no,total):
    p_yes = float(yes / total)
    p_no = float(no / total)
    entropy = - p_yes*math.log(p_yes,2)-p_no*math.log(p_no,2)
    return entropy

def calc_yes_and_no(list_prediction):
    yes = no = 0
    for j in list_prediction:
        if j == 'yes':
            yes += 1
        else:
            no += 1
    return yes,no

def calcAccuracy(test, list_algo):
    for i in range(len(test)):
        if i == len(test) - 1:
            list_prediction = test[i]

    counter = 0
    accuracy = 0
    print(range(len(test[0])))
    for i in range(len(test[0])):
        if list_algo[i] == list_prediction[i]:
            counter += 1
    accuracy = float(counter/len(test[0]))
    print(accuracy)
    return accuracy

def read_file(filename):
    file = open(filename,"r")
    i = 0
    attributes = []
    for line in file:
        if not i:
            i = 1
            # make attributes
            array = line.split('\t')
            for j in range(len(array)):
                if j == len(array)-1:
                    attributes.append(array[j].strip("\n"))
                else:
                    attributes.append(array[j])
            for k in range(len(attributes)):
                attributes[k] = []
        else:
            # fill the lists of the attributes
            values = line.split('\t')
            for i in range(len(values)):
                if i == len(values) - 1:
                    attributes[i].append(values[i].strip("\n"))
                else:
                    attributes[i].append(values[i])
    return attributes

def main_function():
    attributes = read_file("train.txt")
    tests = read_file("test.txt")
    train_data(attributes,tests)

main_function()