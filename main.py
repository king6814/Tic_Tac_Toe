import random
import math


weight_decimal_point_limit=6


class relu():
    @staticmethod
    def loss(value):
        return max(0,value)
    
    @staticmethod
    def gradient(value,sum_z=None):
        if value<0:
            return 0
        else:
            return value

class sigmoid():
    @staticmethod
    def loss(value): 
        return 1 / (1 + math.exp(-value))   #1/(1+e**(-1*value))
    
    @staticmethod
    def gradient(value,sum_z):
        return value * math.exp(-sum_z)/(1+math.exp(-sum_z))**2

class tanh():
    @staticmethod
    def loss(value):
        return math.tanh(value)             #(e**value-e**(-1*value))/(e**value+e**(-1*value))
    
    @staticmethod
    def gradient(value,sum_z):
        return value * 4 / (math.exp(sum_z)+math.exp(-sum_z))**2

    
class mse():
    @staticmethod
    def loss(expect,real):
        return ((expect-real)**2)/2

    @staticmethod
    def gradient(expect,real):
        return expect-real

class ce():
    @staticmethod
    def loss(expect,real):
        return -1*real*math.log(expect)
    
    @staticmethod
    def gradient(expect,real):
        return -1*real/expect


class node():
    def __init__(self,num_next_node):
        
        self.value = 1
        self.node_connection = [ True for i in range(num_next_node) ]
        self.node_weight = [ round(random.random()+0.001,weight_decimal_point_limit)   for i in range(num_next_node) ]


class layers():
    def __init__(self,num_node,activate_function,loss_function,running_weight=1):
        num_node=list(num_node)+[0]

        self.layer=[ [] for _ in range(len(num_node)) ]
        self.activate_function=activate_function
        self.loss_function=loss_function
        self.running_weight=running_weight

        for i in range(len(num_node)-1):
            for w in range(num_node[i]):
                self.layer[i].append(node(num_node[i+1]))
    
    def show_value(self):
        for i in range(len(self.layer)):
            print( [w.value  for w in self.layer[i]])

    def show_weight(self):
        for i in range(len(self.layer)):
            print( [w.node_weight  for w in self.layer[i]])

    def reset_value(self):
        for i in range(len(self.layer)):
            for w in range(len(self.layer[i])):
                self.layer[i][w].value=0

    def save_weight(self,filename='weight_data.txt'):
        with open(filename,'w') as file:
            for i in range(len(self.layer)-2):
                for w,weights in enumerate(self.layer[i]):
                    file.write('/'.join(map(str,weights.node_weight)))
                    if w < len(self.layer[i])-1:
                        file.write(',')
                file.write('\n')
    
    def load_weight(self,filename='weight_data.txt'):
        with open(filename,'r') as file:
            for i in range(len(self.layer)-2):
                for w,weights in enumerate((file.readline()[:-1]).split(',')):
                    weights=list(map(float, weights.split('/')))
                    if len(self.layer[i][w].node_weight) == len(weights):
                        self.layer[i][w].node_weight = weights[:]

    def run(self,input_data):
        
        self.reset_value()

        for i in range(len(self.layer[0])):
            self.layer[0][i].value=input_data[i]
        
        for i in range(len(self.layer)-1):
            for w in self.layer[i]:
                for j,conected in enumerate(w.node_connection):
                    if conected:
                        self.layer[i+1][j].value += w.value * w.node_weight[j]
            
            for w in self.layer[i+1]:
                w.value = self.activate_function.loss(w.value)

    def backpropagation(self,chain_value,layer_index,node_index):
        if layer_index==0:
            return
        
        sum_z=0
        for i in range(len(self.layer[layer_index-1])):
            if self.layer[layer_index-1][i].node_connection[node_index]==True:
                sum_z+= self.layer[layer_index-1][i].value * self.layer[layer_index-1][i].node_weight[node_index]

        for i in range(len(self.layer[layer_index-1])):
            if self.layer[layer_index-1][i].node_connection[node_index]==True:
                self.backpropagation(chain_value * self.activate_function.gradient(self.layer[layer_index-1][i].value,sum_z),  layer_index-1,  i)

        for i in range(len(self.layer[layer_index-1])):
            self.layer[layer_index-1][i].node_weight[node_index]-=self.running_weight*chain_value

    def do_gradient_descent(self,correct_label):
        output_layer_index =len(self.layer)-2

        for i in range(len(self.layer[output_layer_index])):
            chain_value=self.loss_function.gradient(self.layer[output_layer_index][i].value , correct_label[i])
            self.backpropagation(chain_value,output_layer_index,i)



a=layers((9,9,9,9,9),sigmoid,ce)
# a.show_weight()
# a.load_weight()
# a.show_weight()
a.run((1,1,1,1,1,1,1,1,1))
a.show_value()