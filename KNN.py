
# class knn - chosse the k nearest neigbors and classify each of the examples given/
class KNN_algo:

    def __init__(self):
        pass

    # this function calculates the accuracy of the test. it takes the two vectors and
    # in how much examples we were correct due to our model.
    def calcAccuracy(self, test, list_algo):
        for i in range(len(test)):
            if i == len(test) - 1:
                list_prediction = test[i]

        counter = 0
        accuracy = 0
        for i in range(len(test[0])):
            if list_algo[i] == list_prediction[i]:
                counter += 1
        accuracy = float(counter / len(test[0]))
        return accuracy

    # this function calculates the hamming distance for each examples of the test and the
    # train. the distance is calculated by two different strings as 1, and zero for similarity.
    # we loop all the train for each example and take the 5 nearest neigbors.
    def hamming_distance(self, train, test, list_check, list_prediction):
        idx = 0
        list_knn_result = []
        a = len(list_check)
        for k in range(len(list_check[idx])):
            params = []
            while idx < a:
                params.append(list_check[idx][k])
                idx += 1
            idx = 0
            list = []
            counter = 0
            u = 0
            b = len(train)
            for j in range(len(train[u])):
                t_param = []

                while u < a:
                    t_param.append(train[u][j])
                    u += 1
                t_param.append(train[u][j])
                for i in range(a):
                    if params[i] != t_param[i]:
                        counter += 1
                if counter == 0:
                    list.append((counter, t_param[a]))
                else:
                    list.append((counter, t_param[a]))
                counter = 0
                u = 0
            listtry = list

            sort_knn = []
            x = range(len(listtry))
            for j in x:
                m = min(listtry, key=lambda t: t[0])
                sort_knn.append(m)
                listtry.remove(m)
            new_list = [x[1] for x in sort_knn]
            set_labels = set(list_prediction)
            label = []
            for i in set_labels:
                label.append(i)
            label.sort()
            array = [0] * len(label)
            for k in range(5):
                for i in range(len(label)):
                    if new_list[k] == label[i]:
                        array[i] += 1
            maximum = max(array)
            for i in range(len(array)):
                if array[i] == maximum:
                    list_knn_result.append(label[i])
        accuracy = self.calcAccuracy(test, list_knn_result)
        return list_knn_result, accuracy
