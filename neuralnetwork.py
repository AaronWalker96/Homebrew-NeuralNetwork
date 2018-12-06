import random
from math import exp

#Input layer
inputNode1 = 9
inputNode2 = 3

#Random weight generator
def randWeight():
    return random.uniform(0.0, 1.0)

#Weight set 1 (input layer to hidden layer)
w1 = randWeight()
w2 = randWeight()
w3 = randWeight()
w4 = randWeight()
w5 = randWeight()
w6 = randWeight()

#Hidden layer 
hiddenNode1 = 0
hiddenNode2 = 0
hiddenNode3 = 0

#Weight set 2 (hidden layer to output layer)
w7 = randWeight()
w8 = randWeight()
w9 = randWeight()

#Output layer
outputNode = 0

#Calculate the initial hidden node vales
hiddenNode1 = ((inputNode1 * w1) + (inputNode2 * w4))
hiddenNode2 = ((inputNode1 * w2) + (inputNode2 * w5))
hiddenNode3 = ((inputNode1 * w3) + (inputNode2 * w6))

#Apply the activation function to the initial hidden layer value
def sigmoid( x ):
    sx = ( exp(x) / (exp(x) + 1) )
    return sx

hiddenNode1 = sigmoid(hiddenNode1)
hiddenNode2 = sigmoid(hiddenNode2)
hiddenNode3 = sigmoid(hiddenNode3)

#Calculate initial output value 
outputNode = ((hiddenNode1 * w7) + (hiddenNode2 * w8) + (hiddenNode3 * w9))

#Final output value
outputNode = sigmoid(outputNode)

print(outputNode)



