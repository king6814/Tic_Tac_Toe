import random
import math


def relu(value):
    return max(0,value)

def sigmoid(value):
    return 1 / (1 + math.exp(-value))   #1/(1+e**(-1*value))

def tanh(value):
    return math.tanh(value)             #(e**value-e**(-1*value))/(e**value+e**(-1*value))


class node():
    def __init__(self,num_next_node):
        
        self.value = 0
        self.node_connection = [ True for i in range(num_next_node) ]
        self.node_weight = [ random.random()+0.001   for i in range(num_next_node) ]


class layers():
    def __init__(self,*num_node):
        num_node=tuple(list(num_node)+[0])

        self.layer=[ [] for _ in range(len(num_node)) ]

        for i in range(len(num_node)-1):
            for w in range(num_node[i]):
                self.layer[i].append(node(num_node[i+1]))
    
    def show_value(self):
        for i in range(len(self.layer)):
            print( [w.value  for w in self.layer[i]])

    def reset_value(self):
        for i in range(len(self.layer)):
            for w in range(len(self.layer[i])):
                self.layer[i][w].value=0

    def activate(self,input_data,activate_function):
        
        self.reset_value()

        for i in range(len(self.layer[0])):
            self.layer[0][i].value=input_data[i]
        
        for i in range(len(self.layer)-1):
            for w in self.layer[i]:
                for j,conected in enumerate(w.node_connection):
                    if conected:
                        self.layer[i+1][j].value += w.value * w.node_weight[j]
            
            for w in self.layer[i+1]:
                w.value = activate_function(w.value)


a=layers(9,9,9,9,9)
a.show_value()
a.activate((1,2,3,4,5,6,7,8,9),sigmoid)
a.show_value()