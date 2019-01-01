class Node(object):

    def __init__(self, father, type_developed, attribute, values=None):
        if (father != None):
            self.father = father
        else:
            father = None
        if type_developed != None:
            self.type_developed = type_developed
        else:
            type_developed = None
        self.attribute = attribute
        if values is None:
            self.values = []
        else:
            self.values = values
        self.dict = {}
