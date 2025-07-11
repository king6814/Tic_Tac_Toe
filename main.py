import random

class node():
    def __init__(self,num_next_node):
        
        self.value = 0
        self.node_connection = [ True for i in range(num_next_node) ]
        self.node_weight = [ random.random()+0.1   for i in range(num_next_node) ]



class layers():
    def __init__(self,*num_node):
        num_node=tuple(list(num_node)+[0])

        self.layer=[ [] for i in range(len(num_node)) ]

        for i in range(len(num_node)-1):
            for w in range(num_node[i]):
                self.layer[i].append(node(num_node[i+1]))


a=layers(9,8,6,9)
for i in range(5):
    print( [w.value  for w in a.layer[i]])