import pandas as pd
import numpy as np
import math as math
import operator as operator
from DTL import Decition_Tree
def train_data(tags,train,test):
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
    #hamming_distance(train,test,list_check,list_prediction)
    naive_bayes(train,test,list_check,list_prediction)
    #id3(tags,train, test, list_check, list_prediction)

def hamming_distance(train,test,list_check,list_prediction):
    idx = 0
    list_knn_result = []
    print(list_knn_result)
    a = len(list_check)
    for k in range(len(list_check[idx])):
        params = []
        while idx < a:
            params.append(list_check[idx][k])
            idx+=1
        idx =0
        list = []
        counter = 0
        u=0
        b = len(train)
        for j in range(len(train[u])):
            t_param = []

            while u < a:
                t_param.append(train[u][j])
                u += 1
            t_param.append(train[u][j])
            for i in range(a):
                if params[i]!=t_param[i]:
                    counter += 1
            if counter == 0:
                 list.append((counter,t_param[a]))
            else:
                list.append((counter,t_param[a]))
            counter = 0
            u = 0
        list.sort(key=operator.itemgetter(0))

        yes = 0
        no = 0
        new_list = [x[1] for x in list]
        set_labels = set(list_prediction)
        label = []
        for i in set_labels:
            label.append(i)
        for k in range(5):
            if new_list[k] == label[0]:
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
    set_labels = set(list_prediction)
    label = []
    for i in set_labels:
        label.append(i)

    for i in range(len(train)):
        if i == len(train) - 1:
            list_prediction = train[i]
    for j in list_prediction:
        if j == label[0]:
            yes += 1
        else:
            no += 1
    p_train_yes = float(yes/len(train[0]))
    p_train_no = float(no/len(train[0]))
    a = len(list_check)
    for k in range(len(list_check[idx])):
        params = []
        while idx < a:
            params.append(list_check[idx][k])
            idx += 1
        idx = 0
        list = []
        counter_yes = [0] * a
        counter_no = [0] * a
        u = 0
        train_len = len(train[u])
        for j in range(len(train[u])):
            t_param = []
            while u < a:
                t_param.append(train[u][j])
                u += 1
            t_param.append(train[u][j])
            for i in range(a):
                if params[i] == t_param[i]:
                    if t_param[u] == label[0]:
                        counter_yes[i]+=1
                    else:
                        counter_no[i]+=1
            u=0
        prob_yes = [0] * a
        prob_no = [0] * a
        temp_yes = 1
        temp_no = 1
        for i in range(a):
            prob_yes[i] = float(counter_yes[i]/yes)
            prob_no[i] = float(counter_no[i]/no)
        for i in range(a):
            temp_yes*=prob_yes[i]
            temp_no*=prob_no[i]
        p_example_yes = p_train_yes * temp_yes
        p_example_no = p_train_no * temp_no

        if p_example_no > p_example_yes:
            list_naive_bayes_result.append(label[1])
        else:
            list_naive_bayes_result.append(label[0])
    print(list_naive_bayes_result)
    calcAccuracy(test,list_naive_bayes_result)

def id3(tags,train,test,list_check,list_prediction):

    train_without_decision = []
    yes = no = p_train_yes = p_train_no =  0
    for i in range(len(train)):
        if i == len(train) - 1:
            # classify of last column in train
            list_prediction = train[i]
        else:
            train_without_decision.append(train[i])

    yes,no = calc_yes_and_no(list_prediction)
    if yes > no:
        defult = 'yes'
    else:
        defult = 'no'
    len_train = len(train[0])
    dtl_algo(tags, train, defult)
    decision_tree = Decition_Tree(dtl_algo(tags,train,'no'))


def dtl_algo(attributes, examples, defult):
    list_id3 = []
    level = 0
    train_without_decision = []
    for i in range(len(examples)):
        if i == len(examples) - 1:
            # classify of last column in train
            list_prediction = examples[i]
        else:
            train_without_decision.append(examples[i])
    set_decision = set(list_prediction)
    yes, no = calc_yes_and_no(list_prediction)
    if len(examples) == 0:
        if yes> no:
            defult = 'yes'
            return defult
        else:
            defult = 'no'
            return defult
    elif yes == 0:
        defult = 'no'
        return defult
    elif no == 0:
        defult = 'yes'
        return defult
    elif len(set_decision) == 1:
        return set_decision.pop()
    else:
        best,values = choose_attribute(attributes, train_without_decision,list_prediction)
        root = Decition_Tree(best,values,level)
        list_id3.append(root)
        level += 1

        new_attributes = []
        new_examples = []
        j = 0
        for i in attributes:
            if i != best:
                new_attributes.append(i)
            else:
                column = j
            j += 1
        j = 0
        for i in train_without_decision:
            if j != column:
                new_examples.append(i)
            else:
                set = set(examples[column])
        for val in set:
            best,values = dtl_algo(new_attributes,new_examples,defult)
            list_id3.append(Decition_Tree.__init__(best, values, level))
        level +=1
    return best,values

def choose_attribute(attributes, examples,list_prediction):
    optional_attributes = []
    train = examples
    yes,no = calc_yes_and_no(list_prediction)
    decision_entropy = calc_entropy(yes, no, len(train[0]))
    attributes_gain = []
    j = 0
    for i in examples:
        sub_entropy = check_attribute(i, list_prediction)
        attributes_gain.append(decision_entropy - sub_entropy)
        optional_attributes.append((attributes[j],attributes_gain[j]))
        j+=1
    j=0
    optional_attributes.sort(reverse=True,key=operator.itemgetter(1))
    return optional_attributes[j]




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
        tags = []
        for i in range(len(array)):
            if i == len(array)-1:
                tags.append(array[i].strip("\n"))
            else:
                tags.append(array[i])
    return tags,attributes

def main_function():
    tags,attributes = read_file("train.txt")
    tags_test,tests = read_file("test.txt")
    train_data(tags,attributes,tests)

main_function()
