class Node(object):

    def __init__(self, father, type_developed, attribute, values=None):
        if (father != None):
            self.father = father
        else:
            father = None
        if type_developed != None:
            self.type_developed = type_developed
        else:
            self.type_developed = None
        self.attribute = attribute
        if values is None:
            self.values = []
        else:
            self.values = values
        self.dict = {}


class newNode:
  def __init__(self, attName, attClass, dictSubNodes):
    self.attName = attName
    self.attClass = attClass
    self.dictSubnodes = dictSubNodes
  def __init__(self, attClass):
      self.attName = None
      self.attClass = attClass
      self.dictSubnodes = None