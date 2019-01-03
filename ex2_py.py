import pandas as pd
import numpy as np
import math as math
from KNN import KNN_algo
import DTL as dtl
import operator as operator
from DTL import Node
from class_node import ClassNode


def train_data(tags, train, test):
    list_check = []
    for i in range(len(train)):
        if i == len(train) - 1:
            list_prediction = test[i]
        else:
            list_check.append(test[i])
    # train = all data
    # test = all test
    # list check = all test but decision column
    # list predection last column of test, the column to compare to
    # k = KNN_algo()
    # k.hamming_distance(train,test,list_check,list_prediction)
    # naive_bayes(train,test,list_check,list_prediction)
    before_id3(train, test, list_check, list_prediction, tags, defult=0)


# def hamming_distance(train, test, list_check, list_prediction):
#     idx = 0
#     list_knn_result = []
#     # print(list_knn_result)
#     a = len(list_check)
#     for k in range(len(list_check[idx])):
#         params = []
#         while idx < a:
#             params.append(list_check[idx][k])
#             idx += 1
#         idx = 0
#         list = []
#         counter = 0
#         u = 0
#         b = len(train)
#         for j in range(len(train[u])):
#             t_param = []
#
#             while u < a:
#                 t_param.append(train[u][j])
#                 u += 1
#             t_param.append(train[u][j])
#             for i in range(a):
#                 if params[i] != t_param[i]:
#                     counter += 1
#             if counter == 0:
#                 list.append((counter, t_param[a]))
#             else:
#                 list.append((counter, t_param[a]))
#             counter = 0
#             u = 0
#         list.sort(key=operator.itemgetter(0))
#
#         new_list = [x[1] for x in list]
#         set_labels = set(list_prediction)
#         label = []
#         for i in set_labels:
#             label.append(i)
#         label.sort()
#         array = [0] * len(label)
#         for k in range(5):
#             for i in range(len(label)):
#                 if new_list[k] == label[i]:
#                     array[i] += 1
#         maximum = max(array)
#         for i in range(len(array)):
#             if array[i] == maximum:
#                 list_knn_result.append(label[i])
#     calcAccuracy(test, list_knn_result)

def naive_bayes(train, test, list_check, list_prediction):
    list_naive_bayes_result = []
    yes = no = p_train_yes = p_train_no = idx = 0
    set_labels = set(list_prediction)
    label = []
    for i in set_labels:
        label.append(i)
    label.sort()
    for i in range(len(train)):
        if i == len(train) - 1:
            list_prediction = train[i]
    for j in list_prediction:
        if j == label[0]:
            no += 1
        else:
            yes += 1
    p_train_yes = float(yes / len(train[0]))
    p_train_no = float(no / len(train[0]))
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
                    if t_param[u] == label[1]:
                        counter_yes[i] += 1
                    else:
                        counter_no[i] += 1
            u = 0
        prob_yes = [0] * a
        prob_no = [0] * a
        temp_yes = 1
        temp_no = 1
        for i in range(a):
            prob_yes[i] = float(counter_yes[i] / yes)
            prob_no[i] = float(counter_no[i] / no)
        for i in range(a):
            temp_yes *= prob_yes[i]
            temp_no *= prob_no[i]
        p_example_yes = p_train_yes * temp_yes
        p_example_no = p_train_no * temp_no

        if p_example_no > p_example_yes:
            list_naive_bayes_result.append(label[0])
        elif p_example_no == p_example_yes:
            if p_train_no > p_train_yes:
                list_naive_bayes_result.append(label[0])
            else:
                list_naive_bayes_result.append(label[1])
        else:
            list_naive_bayes_result.append(label[1])
    calcAccuracy(test, list_naive_bayes_result)


def before_id3(train, test, list_check, list_prediction, tags, defult):
    d_tree = []
    id3(train, tags, defult, d_tree, None, None, None)
    root = d_tree[0]
    examples = []
    # check the test on the root of the tree
    for i in range(len(list_check[0])):
        variable = [x[i] for x in list_check]
        examples.append(variable)
    # result is the prediction of the decision tree
    result, dict_tags_values, list_tracks = check_test(root, tags, train, examples)
    tracks = ramove_repeat_track(list_tracks)
    printer = print_tracks(d_tree[0], dict_tags_values, 1)
    print(printer)
    calcAccuracy(test, result)
    print()


def print_tracks(root, dict_tags_values, tab):
    printer = ""

    for i, key in enumerate(sorted(root.dict.keys())):
        if i != 0:
            for j in range(tab - 1):
                printer += '\t'
        next_root = root.dict[key]
        if type(next_root.attribute) is str:
            if next_root.attribute == 'no' or next_root.attribute == 'yes':
                printer += "|" + root.attribute[0] + "=" + key + ":" + next_root.attribute + '\n'

        else:
            if tab == 1:
                printer += root.attribute[0] + "=" + key + "\n"
            else:
                printer += "|" + root.attribute[0] + "=" + key + '\n'
            for i in range(tab):
                printer += '\t'
            printer += print_tracks(next_root, dict_tags_values, tab + 1)
    return printer


def ramove_repeat_track(list_tracks):
    tracks = []
    for list in list_tracks:
        if len(tracks) == 0:
            tracks.append(list)
        flag = 0
        for l in tracks:
            for item in range(len(l)):
                if l == list:
                    flag = 1
                    break
        if not flag:
            tracks.append(list)

    return tracks


def check_test(root, tags, train, examples):
    dict_tags_values = {}
    list_values = []
    result = []
    for i in range(len(tags) - 1):
        set_values = set(train[i])
        list_values = set_values
        dict_tags_values[tags[i]] = list_values
    list_attributes = []
    order = get_attributes_order(tags, root, list_attributes)
    fix_examples = []
    for i in range(len(examples)):
        current = examples[i]
        fix = []
        for attribute in order:
            for key, values in dict_tags_values.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
                if key == attribute:
                    for quality in current:
                        for val in values:
                            if quality == val:
                                fix.append(quality)
        fix_examples.append(fix)
    list_tracks = []
    temp_root = root
    prev = temp_root
    for j in range(len(fix_examples)):
        list = []
        current = fix_examples[j]
        temp_root = root
        prev = temp_root
        while any(temp_root.dict) != False:
            list.append(temp_root.attribute[0])

            for i in current:
                list.append(i)
                temp_root = prev.dict[i]
                if temp_root.attribute == 'yes' or temp_root.attribute == 'no':
                    list.append(temp_root.attribute)
                    result.append(temp_root.attribute)
                    break
                else:
                    prev = temp_root
                    list.append(prev.attribute[0])
                    tag = next(iter(temp_root.dict))
                    temp_root = temp_root.dict[tag]
        list_tracks.append(list)
    print(list_tracks)
    return result, dict_tags_values, list_tracks


def get_attributes_order(tags, root, list_attributes):
    temp_root = root
    while temp_root.attribute[0] != None and len(list_attributes) != len(tags) - 1:
        if len(list_attributes) == len(tags) - 1:
            break
        print(temp_root.attribute[0])
        for i in tags:
            if temp_root.attribute[0] == i:
                list_attributes.append(i)
                break
        if len(list_attributes) == len(tags) - 1:
            break
        tag = next(iter(temp_root.dict))
        temp_root = temp_root.dict[tag]
    return list_attributes


def id3(train, tags, defult, tree, father, type_developed, father_node=None):
    train_without_decision = []
    for i in range(len(train)):
        if i == len(train) - 1:
            # classify of last column in train
            list_prediction = train[i]
        else:
            train_without_decision.append(train[i])
    set_labels = set(list_prediction)
    label = []
    decision = calc_predictions(list_prediction)
    maxVal = max(decision)
    for i in set_labels:
        label.append(i)
    label.sort()
    labels_with_nums = []
    for i in range(len(label)):
        labels_with_nums.append((label[i], decision[i]))
        if maxVal == decision[i]:
            defult = label[i]
    current_examples = train

    for i in labels_with_nums:
        if i[1] == 0:
            class_node = ClassNode(i[0], father_node)
            father_node[type_developed] = class_node
            return

    best = choose_attribute(tags, train, list_prediction)
    # לייצר אטרביוט נוד שמכיל את בסט כאטריובט וכנכנסים לרקורסיה להצביע על הנוד אטריביוט הזה
    k = col = 0
    new_tags = []
    labels = []
    for i in tags:
        if best[0] == i:
            values = set(train[k])
            for i in values:
                labels.append(i)
            col = k
        k += 1
    labels.sort()
    node = Node(father, type_developed, best, labels)
    len_labels = len(labels)
    tree.append(node)
    if father is not None:
        father_node.dict[type_developed] = node

    for i in list_prediction:
        if best == i:
            return best
    len_tags = len(tags)
    new_train = []
    for i in range(len(labels)):
        new_train.append([])
    index = 0
    for value in labels:
        for i in range(len_tags):
            new_train[index].append([])
        for i in range(len(train[index])):
            # print(train[col][i])
            if (train[col][i] == value):
                for k in range(len(tags)):
                    # print(train[k][i])
                    new_train[index][k].append(train[k][i])

        index += 1

    # if all columns with the same value return node with  value decision
    for training in new_train:
        set_values = get_values_of_attribute(best, tags, train)
        set_type_developed = set(training[col])
        value = set_type_developed.pop()
        id3(training, tags, defult, tree, best, value, node)


def get_values_of_attribute(best, tags, train):
    for i in range(len(tags)):
        if best == tags[i]:
            return set(train[i])


def choose_attribute(attributes, examples, list_prediction):
    p_predictions = calc_predictions(list_prediction)
    len_sets = []
    for i in range(len(attributes) - 1):
        setI = set(examples[i])
        len_sets.append(len(setI))
    if sum(len_sets) == len(attributes) - 1:
        label = []
        maxVal = max(p_predictions)
        for i in set(list_prediction):
            label.append(i)
        label.sort()
        labels_with_nums = []
        for i in range(len(label)):
            labels_with_nums.append((label[i], p_predictions[i]))
            if maxVal == p_predictions[i]:
                defult = label[i]
                return defult

    entropy = calc_entropy(p_predictions)
    optional_attributes = []
    attributes_gain = []
    j = 0
    for example, i in zip(examples, range(len(examples) - 1)):
        list_att, list_labels = calc_predictions_with_decision(example, list_prediction)
        sumEnt = entropy
        for item in list_att:
            p = sum(item) / len(examples[0])
            sub_entropy = calc_entropy(item)
            sumEnt += -p * sub_entropy
        attributes_gain.append(sumEnt)
        optional_attributes.append((attributes[j], attributes_gain[j]))
        j += 1
    j = 0
    optional_attributes.sort(reverse=True, key=operator.itemgetter(1))

    return optional_attributes[j]


def calc_entropy(p_list):
    entropy = []
    total = sum(p_list)
    sum1 = 0
    for i in range(len(p_list)):
        p = p_list[i] / total
        if p == 0:
            entropy.append(0)
        else:
            entropy.append(-p * math.log(p, 2))
    for j in range(len(entropy)):
        sum1 += entropy[j]
    return sum1


def calc_predictions(list_prediction):
    labels = set(list_prediction)
    list_lables = p_list = []
    for i in labels:
        list_lables.append(i)
    list_lables.sort()
    list = [0] * len(set(list_prediction))
    total = len(list_prediction)
    for j in range(len(list_prediction)):
        for i in range(len(list_lables)):
            if list_prediction[j] == list_lables[i]:
                list[i] += 1
    return list


def calc_predictions_with_decision(list_prediction, decision):
    labels = set(list_prediction)
    decisions = set(decision)
    list_lables, p_list, list_decision = [], [], []
    for i in labels:
        list_lables.append(i)
    list_lables.sort()
    for i in decisions:
        list_decision.append(i)
    list_decision.sort()

    list = [] * len(set(list_prediction))
    for j in range(len(set(list_prediction))):
        list.append([0] * len(decisions))

    # list = [[0] for i in range(len(decisions))] * len(set(list_prediction))
    total = len(list_prediction)
    for j, index in zip(list_prediction, range(len(list_prediction))):
        for i, index2 in zip(list_lables, range(len(list_lables))):
            if j == i:
                for k, index3 in zip(list_decision, range(len(list_decision))):
                    if decision[index] == k:
                        list[index2][index3] += 1
    return list, list_lables


def calc_yes_and_no(list_prediction):
    yes = no = 0
    for j in list_prediction:
        if j == 'yes':
            yes += 1
        else:
            no += 1
    return yes, no


def calcAccuracy(test, list_algo):
    for i in range(len(test)):
        if i == len(test) - 1:
            list_prediction = test[i]

    counter = 0
    accuracy = 0
    # print(range(len(test[0])))
    for i in range(len(test[0])):
        if list_algo[i] == list_prediction[i]:
            counter += 1
    accuracy = float(counter / len(test[0]))
    accuracy = str(round(accuracy, 2))
    print(accuracy)
    return accuracy


def read_file(filename):
    file = open(filename, "r")
    i = 0
    attributes = []
    for line in file:
        if not i:
            i = 1
            # make attributes
            array = line.split('\t')
            for j in range(len(array)):
                if j == len(array) - 1:
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
            if i == len(array) - 1:
                tags.append(array[i].strip("\n"))
            else:
                tags.append(array[i])
    return tags, attributes


def main_function():
    tags, attributes = read_file("train.txt")
    tags_test, tests = read_file("test.txt")
    train_data(tags, attributes, tests)


main_function()
