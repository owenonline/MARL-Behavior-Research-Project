import numpy as np
import matplotlib.pyplot as plot
import os
import numpy as np
import math
from random import randint

#Activation Functions
def sigmoid(X):
    return 1/(1+np.exp(-X))

def relu(X):
    return np.maximum(0,X)

def tanh(X):
    return np.tanh(X)

def softmax(X):
    exp_X = np.exp(X)
    exp_X_sum = np.sum(exp_X,axis=1).reshape(-1,1)
    exp_X = exp_X/exp_X_sum
    return exp_X

def swish(X):
    return X/(1+np.exp(-X))

#derivative of tanh
def tanh_derivative(X):
    return 1-(X**2)

class checkersEnvironment():
    def envInit(self):
        self.terminal=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        
    def envStart(self):
        #reward is a 3x3 matrix. The first row is the message reward (based on how much the messages relate to the actions taken), the second row is the board reward (based on the legal movement of pieces; >=0), the third is general reward (based on winning or losing the game)
        reward=[[0,0,0],[0,0,0],[0,0,0]]
        #messages is a vector for 4 messages of length 0-100. message 1 is from agent 3 to agent 1, message 2 is from agent 3 to agent 2, message 3 is from agent 1 to agent 3, and message 4 is from agent 2 to agent 3
        messages=[list(0 for x in range(100)),list(0 for x in range(100)),list(0 for x in range(100)),list(0 for x in range(100))]
        #relation for agent 1 then 2
        relationVals=[0,0]
        #(y,x)
        boardState=[[[2,1],[1,2],[2,3],[1,4],[2,5],[1,6]],[[6,1],[5,2],[6,3],[5,4],[6,5],[5,6]]]
        isTerminal=False
        self.fullState=(reward,boardState,isTerminal,messages,relationVals)
        return self.fullState
            
    def envStepBoard(self,action,piece,agentReal):
        #agentReal=-1, -2, 1, or 2 to represent either agent 1, agent 2, or either agent but selected by agent 3 (the negative values). This is normalized to a binary value for the purpose of determining move legality.
        agent=abs(agentReal)-1
        #agent: either 0 or 1 to represent agents 1 or 2
        #piece: any integer 0-5 to allow choice of any piece
        #action: [(-2-2), (-2-2), (-2-2), (-2-2), (-2-2), (-2-2)] with those being the movement increments. Any single move in checkers is either 0, 1, or 2 spaces in length hence the (-2-2). a negative value moves left diagonal
        lastState=self.fullState[1] #this gets the state from the tuple
        pieceStart=lastState[agent][piece] #this gets the situation of the piece the agent chose to move as of the last state.  [row,column]          

        currentState=lastState
        boardReward=0
        generalReward=0
        isTerminal=False

        for x in range(len(action)):
            if pieceStart==[0,0]:
                #this means an already eliminated piece has been selected
                boaordReward+=-10
                break
            
            #checks if the move was legal
            if action[x]==0:
                if x==0:
                    #not moving any piece is an illegal move
                    boardReward+=-10
                    break

            #checks that the new index is within the board and is unoccupied by either an opposing or friendly piece
            elif 1<=pieceStart[0]+abs(action[x])<6 and 1<pieceStart[1]+action[x]<=6 and [pieceStart[0]+abs(action[x]),pieceStart[1]+action[x]] not in lasteState[0] and [pieceStart[0]+abs(action[x]),pieceStart[1]+action[x]] not in lastState[1]:
                #if no move to capture was made and the previous check was passed, the move is legal and the current state can be updated to reflect that.
                if action[x]==1 or action[x]==-1:
                    currentState[agent][piece]=[pieceStart[0]+abs(action[x]),pieceStart[1]+action[x]]
                    break
                elif action[x]==2 or action[x]==-2:
                    #this just checks if the jumped over square contains an enemy piece. this must be true for the move to be legal
                    if [pieceStart[0]+abs(action[x])-1,pieceStart[1]+((abs(action[x])-1)*(action[x]/abs(action[x])))] in lastState[-1*agent+1]:
                        currentState[agent][piece]=[pieceStart[0]+abs(action[x]),pieceStart[1]+action[x]]
                        #sets the state of the enemy piece that was taken to 0
                        currentState[-1*agent+1][lastState[-1*agent+1].index([pieceStart[0]+abs(action[x])-1,pieceStart[1]+((abs(action[x])-1)*(action[x]/abs(action[x])))])]=[0,0]
                        #sets piecestart for the starting position of the next part of the move
                        pieceStart=currentState[agent][piece]
                    else:
                        #the agent used a jump move without eliminating an enemy piece, which is an illegal move
                        boardReward+=-10
                        break
                else:
                    #invalid move length
                    boardReward+=-10
                    break
            else:
                #this means the move selected is illegal because it lands on another piece or goes off the board
                boardReward+=-10
                break
            

        if (currentState[0]==self.terminal and agentReal==1) or (currentState[1]==self.terminal and agentReal==2):
            #agents 1 and 2 get a -100 general reward for losing
            generalReward=-100
            isTerminal=True
        elif (currentState[1]==self.terminal and agentReal==1) or (currentState[0]==self.terminal and agentReal==2):
            #agents 1 and 2 get a 100 general reward for winning
            generalReward=100
            isTerminal=True
        elif (agentReal==-1 or agentReal==-2) and (currentState[0]==self.terminal or currentState[1]==self.terminal):
            #agent 3 gets a -100 general reward when the game ends
            generalReward=-100
            isTerminal=True
        elif (agentReal==-1 or agentReal==-2) and not (currentState[0]==self.terminal or currentState[1]==self.terminal):
            #agent 3 gets a +1 general reward every time step
            reward=1
        elif (agentReal==1 or agentReal==2) and not (currentState[0]==self.terminal or currentState[1]==self.terminal):
            #agents 1 and 2 get a -1 general reward every time step
            reward=-1

        return (currentState, isTerminal, boardReward, generalReward)


class agentThree():
    def __init__(self):
        #each lstm has 4 weights and 4 biases see https://www.kaggle.com/navjindervirdee/lstm-neural-network-from-scratch and http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        #lstm params has msg1-4 then board params in the order fa, ia, ga, oa with bias params
        #lstm backprop state has msg1-4 then board in the order fa, ia, ga, oa for back propogation
        #lstm recursive info has msg1-4 then board with the cell state then the last activation matrix
        #100 input length, 300 hidden length, so 400 total input length
        self.lstm_params=[[np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400))],[np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400))],[np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400))],[np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400))],[np.random.normal(0,0.1,(200,224)),np.random.normal(0,0.1,(200,224)),np.random.normal(0,0.1,(200,224)),np.random.normal(0,0.1,(200,224))],[np.random.normal(0,0.1,(50,52)),np.random.normal(0,0.1,(50,52)),np.random.normal(0,0.1,(50,52)),np.random.normal(0,0.1,(50,52))]]
        self.lstm_backprop_state=[[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
        self.lstm_recursive_info=[[np.ones(300),np.ones(300)],[np.ones(300),np.ones(300)],[np.ones(300),np.ones(300)],[np.ones(300),np.ones(300)],[np.ones(200),np.ones(200)],[np.ones(50),np.ones(50)]]
        self.lstm_biases=[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
        
        #ffn is gonna have 2 layers each with relu that cut the length down. 1450->725->363
        self.ffn_params=[np.random.normal(0,0.1,(725,1450)),np.random.normal(0,0.1,(363,725))]
        self.ffn_biases=[0,0]
        self.ffn_outputs=[[],[]]
        
        self.value_fn_params=[]
        
        self.board_policy_params=[np.random.normal(0,0.1,(318,363)),np.random.normal(0,0.1,(273,318)),np.random.normal(0,0.1,(228,273)),np.random.normal(0,0.1,(183,228)),np.random.normal(0,0.1,(138,183)),np.random.normal(0,0.1,(93,138)),np.random.normal(0,0.1,(48,93)),np.random.normal(0,0.1,(8,48))]
        self.board_policy_biases=[0,0,0,0,0,0,0,0,0]
        self.board_policy_backprop_info=[]
        
        self.msg_policy_params=[np.random.normal(0,0.1,(343,363)),np.random.normal(0,0.1,(323,343)),np.random.normal(0,0.1,(303,323)),np.random.normal(0,0.1,(283,303)),np.random.normal(0,0.1,(243,283)),np.random.normal(0,0.1,(223,243)),np.random.normal(0,0.1,(203,223)),np.random.normal(0,0.1,(200,203))]
        self.msg_policy_biases=[0,0,0,0,0,0,0,0,0]
        self.msg_policy_backprop_info=[]
        
    def stateConcatThree(fullState):
        reward=fullState[0]
        boardState=fullState[1]
        temp=[]
        for x in boardState:
            for y in x:
                for z in y:
                    temp.append(z)
        boardState=temp
        isTerminal=fullState[2]
        messageOne=fullState[3][0]
        messageTwo=fullState[3][1]
        messageThree=fullState[3][2]
        messageFour=fullState[3][3]
        relationVals=fullState[4]
        lstm_inputs=[messageOne,messageTwo,messageThree,boardState,relationVals]

        for x in range(len(lstm_inputs)):
            concat_input=np.concatenate((lstm_inputs[x],self.lstm_recursive_info[x][1]))

            fa=sigmoid(np.matmul(self.lstm_params[x][0],concat_input)+self.lstm_biases[x][0])

            ia=sigmoid(np.matmul(self.lstm_params[x][1],concat_input)+self.lstm_biases[x][1])

            ga=tanh(np.matmul(self.lstm_params[x][2],concat_input)+self.lstm_biases[x][2])

            oa=sigmoid(np.matmul(self.lstm_params[x][3],concat_input)+self.lstm_biases[x][3])

            ct=np.multiply(fa,self.lstm_recursive_info[x][0])+np.multiply(ia,ga)

            am=np.multiply(oa, tanh(ct))

            self.lstm_backprop_state[x][0]=fa
            self.lstm_backprop_state[x][1]=ia
            self.lstm_backprop_state[x][2]=ga
            self.lstm_backprop_state[x][3]=oa
            self.lstm_recursive_info[x][0]=ct
            self.lstm_recursive_info[x][1]=am

        absolute_state=np.concatenate((self.lstm_recursive_info[0][1],self.lstm_recursive_info[1][1],self.lstm_recursive_info[2][1],self.lstm_recursive_info[3][1],self.lstm_recursive_info[4][1],self.lstm_recursive_info[5][1]))
        self.ffn_outputs[0]=relu(np.matmul(self.ffn_params[0],absolute_state)+self.ffn_biases[0])
        self.ffn_outputs[1]=relu(np.matmul(self.ffn_params[1],self.ffn_outputs[0])+self.ffn_biases[1])
        #ffn_2 is the networks perception of the state distilled from memory and feed forward layers
        return self.ffn_outputs[1]

    def boardAction(self):
        #363->318->273->228->183->138->93->48->8
        nn1=swish(np.matmul((self.board_policy_params[0],self.ffn_outputs[1])))
        nn2=swish(np.matmul((self.board_policy_params[1],nn1)))
        nn3=swish(np.matmul((self.board_policy_params[2],nn2)))
        nn4=swish(np.matmul((self.board_policy_params[3],nn3)))
        nn5=swish(np.matmul((self.board_policy_params[4],nn4)))
        nn6=swish(np.matmul((self.board_policy_params[5],nn5)))
        nn7=swish(np.matmul((self.board_policy_params[6],nn6)))
        nn8=swish(np.matmul((self.board_policy_params[7],nn7)))
        orig_nn8=nn8
        if nn8[0]>-1.5:
            nn8[0]=-1
        else:
            nn8[0]=-2
        nn8[1]=np.round((5/(1+np.exp(-nn8[1]-5))))
        for x in range(len(nn8[2:])):
            nn8[x+2]=np.round((4/(1+np.exp(-(nn8[1]-4))))-2)
        boardAction=[nn8[0],nn8[1],nn8[2:]]
        self.board_policy_backprop_info=[nn1,nn2,nn3,nn4,nn5,nn6,nn7,orig_nn8]
        return boardAction

    def messageAction(self):
        #363->343->323->303->283->243->223->203->200
        nn1=swish(np.matmul((self.msg_policy_params[0],self.ffn_outputs[1])))
        nn2=swish(np.matmul((self.msg_policy_params[1],nn1)))
        nn3=swish(np.matmul((self.msg_policy_params[2],nn2)))
        nn4=swish(np.matmul((self.msg_policy_params[3],nn3)))
        nn5=swish(np.matmul((self.msg_policy_params[4],nn4)))
        nn6=swish(np.matmul((self.msg_policy_params[5],nn5)))
        nn7=swish(np.matmul((self.msg_policy_params[6],nn6)))
        nn8=swish(np.matmul((self.msg_policy_params[7],nn7)))
        message1=nn8[0:99]
        message2=nn8[100:199]
        messageAction=[message1,message2]
        return messageAction
        
        
class agentTwo():
    def __init__(self):
        #each lstm has 4 weights and 4 biases see https://www.kaggle.com/navjindervirdee/lstm-neural-network-from-scratch and http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        #lstm params has msg1-4 then board params in the order fa, ia, ga, oa with bias params
        #lstm backprop state has msg1-4 then board in the order fa, ia, ga, oa for back propogation
        #lstm recursive info has msg1-4 then board with the cell state then the last activation matrix
        #100 input length, 300 hidden length, so 400 total input length
        self.lstm_params=[[np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400))],[np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400))],[np.random.normal(0,0.1,(200,224)),np.random.normal(0,0.1,(200,224)),np.random.normal(0,0.1,(200,224)),np.random.normal(0,0.1,(200,224))],[np.random.normal(0,0.1,(50,51)),np.random.normal(0,0.1,(50,51)),np.random.normal(0,0.1,(50,51)),np.random.normal(0,0.1,(50,51))]]
        self.lstm_backprop_state=[[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
        self.lstm_recursive_info=[[np.ones(300),np.ones(300)],[np.ones(300),np.ones(300)],[np.ones(200),np.ones(200)],[np.ones(50),np.ones(50)]]
        self.lstm_biases=[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]

        #ffn is gonna have 2 layers each with relu that cut the length down. 850->425->200
        self.ffn_params=[np.random.normal(0,0.1,(425,850)),np.random.normal(0,0.1,(200,425))]
        self.ffn_biases=[0,0]
        self.ffn_outputs=[[],[]]
        
        self.value_fn_params=[]
        self.board_policy_params=[]
        self.msg_policy_params=[]

        
    def stateConcatTwo(fullState):
        reward=fullState[0]
        boardState=fullState[1]
        temp=[]
        for x in boardState:
            for y in x:
                for z in y:
                    temp.append(z)
        boardState=temp
        isTerminal=fullState[2]
        messageTwo=fullState[3][1]
        messageFour=fullState[3][3]
        relationVal=fullState[4][1]
        lstm_inputs=[messageTwo,messageFour,boardState,relationVal]

        for x in range(len(lstm_inputs)):
            concat_input=np.concatenate((lstm_inputs[x],self.lstm_recursive_info[x][1]))

            fa=sigmoid(np.matmul(self.lstm_params[x][0],concat_input)+self.lstm_biases[x][0])

            ia=sigmoid(np.matmul(self.lstm_params[x][1],concat_input)+self.lstm_biases[x][1])

            ga=tanh(np.matmul(self.lstm_params[x][2],concat_input)+self.lstm_biases[x][2])

            oa=sigmoid(np.matmul(self.lstm_params[x][3],concat_input)+self.lstm_biases[x][3])

            ct=np.multiply(fa,self.lstm_recursive_info[x][0])+np.multiply(ia,ga)

            am=np.multiply(oa, tanh(ct))

            self.lstm_backprop_state[x][0]=fa
            self.lstm_backprop_state[x][1]=ia
            self.lstm_backprop_state[x][2]=ga
            self.lstm_backprop_state[x][3]=oa
            self.lstm_recursive_info[x][0]=ct
            self.lstm_recursive_info[x][1]=am

        absolute_state=np.concatenate((self.lstm_recursive_info[0][1],self.lstm_recursive_info[1][1],self.lstm_recursive_info[2][1],self.lstm_recursive_info[3][1]))
        self.ffn_outputs[0]=relu(np.matmul(self.ffn_params[0],absolute_state)+self.ffn_biases[0])
        self.ffn_outputs[1]=relu(np.matmul(self.ffn_params[1],self.ffn_outputs[0])+self.ffn_biases[1])
        #ffn_2 is the networks perception of the state distilled from memory and feed forward layers
        return self.ffn_outputs[1]

            
class agentOne():
    def __init__(self):
        #each lstm has 4 weights and 4 biases see https://www.kaggle.com/navjindervirdee/lstm-neural-network-from-scratch and http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        #lstm params has msg1-4 then board params in the order fa, ia, ga, oa with bias params
        #lstm backprop state has msg1-4 then board in the order fa, ia, ga, oa for back propogation
        #lstm recursive info has msg1-4 then board with the cell state then the last activation matrix
        #100 input length, 300 hidden length, so 400 total input length
        self.lstm_params=[[np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400))],[np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400)),np.random.normal(0,0.1,(300,400))],[np.random.normal(0,0.1,(200,224)),np.random.normal(0,0.1,(200,224)),np.random.normal(0,0.1,(200,224)),np.random.normal(0,0.1,(200,224))],[np.random.normal(0,0.1,(50,51)),np.random.normal(0,0.1,(50,51)),np.random.normal(0,0.1,(50,51)),np.random.normal(0,0.1,(50,51))]]
        self.lstm_backprop_state=[[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
        self.lstm_recursive_info=[[np.ones(300),np.ones(300)],[np.ones(300),np.ones(300)],[np.ones(200),np.ones(200)],[np.ones(50),np.ones(50)]]
        self.lstm_biases=[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]

        #ffn is gonna have 2 layers each with relu that cut the length down. 850->425->200
        self.ffn_params=[np.random.normal(0,0.1,(425,850)),np.random.normal(0,0.1,(200,425))]
        self.ffn_biases=[0,0]
        self.ffn_outputs=[[],[]]
        
        self.value_fn_params=[]
        self.board_policy_params=[]
        self.msg_policy_params=[]

        
    def stateConcatOne(fullState):
        reward=fullState[0]
        boardState=fullState[1]
        temp=[]
        for x in boardState:
            for y in x:
                for z in y:
                    temp.append(z)
        boardState=temp
        isTerminal=fullState[2]
        messageOne=fullState[3][0]
        messageThree=fullState[3][2]
        relationVal=fullState[4][0]
        lstm_inputs=[messageOne,messageThree,boardState,relationVal]

        for x in range(len(lstm_inputs)):
            concat_input=np.concatenate((lstm_inputs[x],self.lstm_recursive_info[x][1]))

            fa=sigmoid(np.matmul(self.lstm_params[x][0],concat_input)+self.lstm_biases[x][0])

            ia=sigmoid(np.matmul(self.lstm_params[x][1],concat_input)+self.lstm_biases[x][1])

            ga=tanh(np.matmul(self.lstm_params[x][2],concat_input)+self.lstm_biases[x][2])

            oa=sigmoid(np.matmul(self.lstm_params[x][3],concat_input)+self.lstm_biases[x][3])

            ct=np.multiply(fa,self.lstm_recursive_info[x][0])+np.multiply(ia,ga)

            am=np.multiply(oa, tanh(ct))

            self.lstm_backprop_state[x][0]=fa
            self.lstm_backprop_state[x][1]=ia
            self.lstm_backprop_state[x][2]=ga
            self.lstm_backprop_state[x][3]=oa
            self.lstm_recursive_info[x][0]=ct
            self.lstm_recursive_info[x][1]=am

        absolute_state=np.concatenate((self.lstm_recursive_info[0][1],self.lstm_recursive_info[1][1],self.lstm_recursive_info[2][1],self.lstm_recursive_info[3][1]))
        self.ffn_outputs[0]=relu(np.matmul(self.ffn_params[0],absolute_state)+self.ffn_biases[0])
        self.ffn_outputs[1]=relu(np.matmul(self.ffn_params[1],self.ffn_outputs[0])+self.ffn_biases[1])
        #ffn_2 is the networks perception of the state distilled from memory and feed forward layers
        return self.ffn_outputs[1]


def main():
    checkers=checkersEnvironment()
    agentOne=agentOne()
    agentTwo=agentTwo()
    agentThree=agentThree()
    
    checkers.envInit()
    fullState=checkers.envStart()
    
    absoluteState3=agentThree.stateConcatThree(fullState)
    absoluteState2=agentTwo.stateConcatTwo(fullState)
    absoluteState1=agentOne.stateConcatOne(fullState)

    boardAction=agentThree.boardAction()
    messageAction=agentThree.messageAction()

main()
    

