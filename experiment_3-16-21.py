import numpy as np
import matplotlib.pyplot as plot
import os
import numpy as np
import math
from random import randint
import csv

#hyperparameters
#same for all 3 agents: 
#starting input range=0,0.1
#starting biases=1 for lstm, 0 for everything else
#discount factor for relation stuff: 0.1
#value discount factor=0.1 (doesn't appear used)
#step size for value function=0.1
#step size for board= 0.1
#step size for messages=0.1
#i starting value=1
#ffn learning rate=0.1
#lstm learning rate=0.1

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

def flatten(matrix):
    try:
        while matrix.ndim!=1:
            matrix=matrix.flatten()
    except:
        matrix=matrix[0]
        while matrix.ndim!=1:
            matrix=matrix.flatten()
    return matrix

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
            
    def envStepBoard(self,boardAction):
        agentReal=boardAction[0]
        piece=int(boardAction[1])
        action=boardAction[2:]
        agent=int(abs(agentReal)-1)
        #agentReal=-1, -2, 1, or 2 to represent either agent 1, agent 2, or either agent but selected by agent 3 (the negative values). This is normalized to a binary value for the purpose of determining move legality.
        #agent: either 0 or 1 to represent agents 1 or 2
        #piece: any integer 0-5 to allow choice of any piece
        #action: [(-2-2), (-2-2), (-2-2), (-2-2), (-2-2), (-2-2)] with those being the movement increments. Any single move in checkers is either 0, 1, or 2 spaces in length hence the (-2-2). a negative value moves left diagonal
        #this gets the state from the tuple
        lastState=self.fullState[1] 

        currentState=lastState#sets the current state to the most updated state before the loop
        boardReward=0
        generalReward=0
        isTerminal=False
        
        for x in range(len(action)):
            pieceStart=currentState[int(agent)][int(piece)] #this gets the situation of the piece at the start of the loop  
            currentAction=int(action[0][x])

            if pieceStart==[0,0]:
                #this means an already eliminated piece has been selected
                boardReward+=-10
                break
            
            #checks if the move was legal
            if currentAction==0:
                if x==0:
                    #not moving any piece on the first move of the turn is illegal
                    boardReward+=-10
                #a move of 0 always ends the turn
                break
                
            
            #checks that the new index is within the board and is unoccupied by either an opposing or friendly piece
            elif 1<=pieceStart[0]+abs(currentAction)<6 and 1<pieceStart[1]+currentAction<=6 and [pieceStart[0]+abs(currentAction),pieceStart[1]+currentAction] not in currentState[0] and [pieceStart[0]+abs(currentAction),pieceStart[1]+currentAction] not in currentState[1]:
                #if no move to capture was made and the previous check was passed, the move is legal and the current state can be updated to reflect that.
                if currentAction==1 or currentAction==-1:
                    currentState[agent][piece]=[pieceStart[0]+abs(currentAction),pieceStart[1]+currentAction]
                    break
                elif currentAction==2 or currentAction==-2:
                    #this just checks if the jumped over square contains an enemy piece. this must be true for the move to be legal
                    if [pieceStart[0]+abs(currentAction)-1,pieceStart[1]+((abs(currentAction)-1)*(currentAction/abs(currentAction)))] in lastState[-1*agent+1]:
                        currentState[agent][piece]=[pieceStart[0]+abs(currentAction),pieceStart[1]+currentAction]
                        #sets the state of the enemy piece that was taken to 0
                        currentState[-1*agent+1][lastState[-1*agent+1].index([pieceStart[0]+abs(currentAction)-1,pieceStart[1]+((abs(currentAction)-1)*(currentAction/abs(currentAction)))])]=[0,0]
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
            generalReward=1
        elif (agentReal==1 or agentReal==2) and not (currentState[0]==self.terminal or currentState[1]==self.terminal):
            #agents 1 and 2 get a -1 general reward every time step
            generalReward=-1

        return (currentState, isTerminal, boardReward, generalReward)

    def envStepMessage(self,stateHistory,messageHistory,actionHistory,externalMessageHistory):
        #lambdaVar is a hyperparameter to weight the error
        lambdaVar=10
        totalSpeakingReward=0
        for x in range(len(stateHistory)-1):
            msgDiff=np.absolute(np.subtract(messageHistory[-1],messageHistory[x])).mean()
            stateDiff=np.absolute(np.subtract(stateHistory[-1],stateHistory[x])).mean()

            msgDiff=1/(1-np.exp(-(msgDiff-4)))
            stateDiff=1/(1-np.exp(-(stateDiff-4)))

            temp=0
            if msgDiff>stateDiff:
                temp=msgDiff/stateDiff
            else:
                temp=stateDiff/msgDiff

            totalSpeakingReward=totalSpeakingReward+(-np.absolute(temp)*lambdaVar)
        totalSpeakingReward=totalSpeakingReward/len(stateHistory)
        #now it's time to get the listening reward.
        totalListeningReward=0
        for x in range(len(externalMessageHistory)-1):
            actionDiff=np.absolute(np.subtract(actionHistory[-1],actionHistory[x])).mean()
            msgDiff=np.absolute(np.subtract(externalMessageHistory[-1],externalMessageHistory[x])).mean()

            actionDiff=1/(1-np.exp(-(actionDiff-4)))
            msgDiff=1/(1-np.exp(-(msgDiff-4)))

            temp=0
            if msgDiff>actionDiff:
                temp=msgDiff/actionDiff
            else:
                temp=actionDiff/msgDiff

            totalListeningReward=totalListeningReward+(-np.absolute(temp)*lambdaVar)
        totalListeningReward=totalListeningReward/len(externalMessageHistory)
        #to prevent it from being 0 at the start and give a lil motivation imma make it slightly negative
        totalListeningReward=totalListeningReward-0.001
        totalSpeakingReward=totalSpeakingReward-0.001
        return (totalSpeakingReward, totalListeningReward)

    def prepNewState(self,messageAction,isTerminalNew,rewardNew,currentState,absoluteState1,absoluteState2,absoluteState3,agent):
        [reward,boardState,isTerminal,messages,relationVals]=self.fullState
        boardState=currentState
        isTerminal=isTerminalNew
        [a,b,c,d]=messages
        if agent==3:
            reward[2]=rewardNew
            [a,b]=messageAction
        if agent==1:
            reward[0]=rewardNew
            c=messageAction
        if agent==2:
            reward[1]=rewardNew
            d=messageAction
        messages=[a,b,c,d]
        self.fullState=[reward,boardState,isTerminal,messages,relationVals]
        oldAbsoluteState1=absoluteState1
        oldAbsoluteState2=absoluteState2
        oldAbsoluteState3=absoluteState3
        return (oldAbsoluteState1,oldAbsoluteState2,oldAbsoluteState3, self.fullState)

    def updateState(self, relationValues):
        newVals=[]
        for y in relationValues:
            newVals.append(y[0])
        self.fullState[-1]=newVals


class agentThree():
    def __init__(self):
        #each lstm has 4 weights and 4 biases see https://www.kaggle.com/navjindervirdee/lstm-neural-network-from-scratch and http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        #lstm params has msg1-4 then board params in the order fa, ia, ga, oa with bias params
        #lstm backprop state has msg1-4 then board in the order fa, ia, ga, oa for back propogation
        #lstm recursive info has msg1-4 then board with the cell state then the last activation matrix
        #100 input length, 300 hidden length, so 400 total input length
        self.lstm_params=[[np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400))],[np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400))],[np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400))],[np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400))],[np.random.normal(0,0.01,(200,224)),np.random.normal(0,0.01,(200,224)),np.random.normal(0,0.01,(200,224)),np.random.normal(0,0.01,(200,224))],[np.random.normal(0,0.01,(50,52)),np.random.normal(0,0.01,(50,52)),np.random.normal(0,0.01,(50,52)),np.random.normal(0,0.01,(50,52))]]
        self.lstm_backprop_state=[[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
        self.lstm_recursive_info=[[[np.ones(300)],[np.ones(300)]],[[np.ones(300)],[np.ones(300)]],[[np.ones(300)],[np.ones(300)]],[[np.ones(300)],[np.ones(300)]],[[np.ones(200)],[np.ones(200)]],[[np.ones(50)],[np.ones(50)]]]
        self.lstm_biases=[[np.zeros(300),np.zeros(300),np.zeros(300),np.zeros(300)],[np.zeros(300),np.zeros(300),np.zeros(300),np.zeros(300)],[np.zeros(300),np.zeros(300),np.zeros(300),np.zeros(300)],[np.zeros(300),np.zeros(300),np.zeros(300),np.zeros(300)],[np.zeros(200),np.zeros(200),np.zeros(200),np.zeros(200)],[np.zeros(50),np.zeros(50),np.zeros(50),np.zeros(50)]]
                
        #ffn is gonna have 2 layers each with relu that cut the length down. 1450->725->363
        self.ffn_params=[np.random.normal(0,0.01,(725,1450)),np.random.normal(0,0.01,(363,725))]
        self.ffn_biases=[np.zeros(725),np.zeros(363)]
        self.ffn_outputs=[[],[]]
        
        self.value_fn_params=np.random.normal(0,0.01,(1,363))

        #mean and standard deviation
        self.board_policy_params=np.random.normal(0,0.01,(8,363))
        self.board_policy_std_params=np.random.normal(0,0.01,(8,363))
        self.board_policy_biases=np.zeros(8)
        self.board_policy_std_biases=np.zeros(8)
        self.board_policy_backprop_info=[np.zeros(8)]
        
        self.msg_policy_params=np.random.normal(0,0.01,(200,363))
        self.msg_policy_std_params=np.random.normal(0,0.01,(200,363))
        self.msg_policy_biases=np.zeros(200)
        self.msg_policy_std_biases=np.zeros(200)
        self.msg_policy_backprop_info=[np.zeros(200)]

        #the following three lists hold first every absolute state that the agent takes an action from, second every message the agent generates, and third every action the agent takes. They go by time step
        self.absolute_state_history=[]
        self.message_history_self=[np.zeros(200)]
        self.action_history_self=[np.zeros(8)]

        #this holds the messages of the other agents
        self.external_message_history=[]

        self.boardReward_history=[]
        self.generalReward_history=[]
        self.totalSpeakingReward_history=[]
        self.totalListeningReward_history=[]

        self.I=1
        #used for message reward
        self.msgStateHistory=[]

        self.lstm_inputs=[]
        
    def stateConcatThree(self, fullState, msg):
        reward=fullState[2]
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
        x=0
        for y in self.external_message_history:
            if (y==np.concatenate((messageThree,messageFour))).all():
                x=1
        if x==0 and msg!=1:
            self.external_message_history.append(np.concatenate((messageThree,messageFour)))
        lstm_inputs=[messageOne,messageTwo,messageThree,messageFour,boardState,relationVals]
        self.lstm_inputs.append(lstm_inputs)
        for x in range(len(lstm_inputs)):
            concat_input=np.concatenate((lstm_inputs[x],self.lstm_recursive_info[x][1][-1]))
            
            fa=sigmoid(np.matmul(self.lstm_params[x][0],concat_input)+self.lstm_biases[x][0])

            ia=sigmoid(np.matmul(self.lstm_params[x][1],concat_input)+self.lstm_biases[x][1])

            ga=tanh(np.matmul(self.lstm_params[x][2],concat_input)+self.lstm_biases[x][2])

            oa=sigmoid(np.matmul(self.lstm_params[x][3],concat_input)+self.lstm_biases[x][3])

            ct=np.multiply(fa,self.lstm_recursive_info[x][0][-1])+np.multiply(ia,ga)

            am=np.multiply(oa, tanh(ct))
            
            self.lstm_backprop_state[x][0].append(fa)
            self.lstm_backprop_state[x][1].append(ia)
            self.lstm_backprop_state[x][2].append(ga)
            self.lstm_backprop_state[x][3].append(oa)
            self.lstm_recursive_info[x][0].append(ct)
            self.lstm_recursive_info[x][1].append(am)

        absolute_state=np.concatenate((self.lstm_recursive_info[0][1][-1],self.lstm_recursive_info[1][1][-1],self.lstm_recursive_info[2][1][-1],self.lstm_recursive_info[3][1][-1],self.lstm_recursive_info[4][1][-1],self.lstm_recursive_info[5][1][-1]))
        
        self.ffn_outputs[0]=relu(np.matmul(self.ffn_params[0],absolute_state)+self.ffn_biases[0])
        self.ffn_outputs[1]=relu(np.matmul(self.ffn_params[1],self.ffn_outputs[0])+self.ffn_biases[1])
        #ffn_2 is the networks perception of the state distilled from memory and feed forward layers. This also adds the state to the list of every absolute state; len 363
        self.absolute_state_history.append([absolute_state,self.ffn_outputs[0],self.ffn_outputs[1]])
        return self.ffn_outputs[1]

    def boardAction(self):
        nn8=np.matmul(self.board_policy_params,self.ffn_outputs[1])+self.board_policy_biases
        alternateAction=[(np.random.random()*3)-3,(np.random.random()*10)-10,(np.random.random()*8),(np.random.random()*8),(np.random.random()*8),(np.random.random()*8),(np.random.random()*8),(np.random.random()*8)]
        orig_action=[]
        action=[]
        if np.random.randint(1,11)<6:
            orig_action=alternateAction
            action=alternateAction
        else:
            orig_action=nn8
            action=nn8
        
        if action[0]>-1.5:
            action[0]=-1
        else:
            action[0]=-2
        action[1]=np.round((5/(1+np.exp(-action[1]-5))))
        for x in range(len(action[2:])):
            action[x+2]=np.round((4/(1+np.exp(-(action[x+2]-4))))-2)

        boardAction=[action[0],action[1],action[2:]]
        self.board_policy_backprop_info.append(orig_action)
        self.action_history_self.append(action)
        return boardAction

    def messageAction(self):
        nn8=np.matmul(self.msg_policy_params,self.ffn_outputs[1])+self.msg_policy_biases
        message1=nn8[0:100]
        message2=nn8[100:200]
        messageAction=[message1,message2]
        self.msg_policy_backprop_info.append(nn8)
        self.message_history_self.append(nn8)
        self.msgStateHistory.append(self.ffn_outputs[1])
        return messageAction

    def updateParameters(self,oldAbsoluteState3,absoluteState3,currentStateValue1,currentStateValue2,oldStateValue1,oldStateValue2,reward):
        ####This code adds the relation values to the general reward####
        (boardReward,generalReward,totalSpeakingReward,totalListeningReward)=reward
        relations=checkers.fullState[-1]
        #relation discounting amount
        discountFactor=0.1
        relations[0]+=(currentStateValue1-oldStateValue1)*discountFactor
        relations[1]+=(currentStateValue2-oldStateValue2)*discountFactor
        if (relations[0]-checkers.fullState[-1][0]) < (relations[0]-checkers.fullState[-1][0]):
            generalReward+=relations[0]-checkers.fullState[-1][0]
        else:
            generalReward+=relations[1]-checkers.fullState[-1][1]

        ####This code regularizes the reward:####
        self.boardReward_history.append(boardReward)
        self.generalReward_history.append(generalReward)
        self.totalSpeakingReward_history.append(totalSpeakingReward)
        self.totalListeningReward_history.append(totalListeningReward)

        ####This code amalgamates the reward into one value:####
        totalReward=boardReward+generalReward+totalSpeakingReward+totalListeningReward
        totalBoardReward=boardReward+generalReward
        totalMesagReward=generalReward+totalSpeakingReward+totalListeningReward
        ##This code calculates the value of the current state
        currentStateValue3=swish(np.matmul(self.value_fn_params,absoluteState3))
        oldStateValue3=swish(np.matmul(self.value_fn_params,oldAbsoluteState3))

        
        ####This code calculates the value function error, which is important in getting the error of the other states. It also updates the value function####
        valueDiscount=0.1
        stepSizeValue=0.01
        #y'=y+sigmoid(x)*(1-y)...this is multiplied by the old state because to do the backprop thing u have to back propogate all the way from the value to the input
        value_derivative=oldStateValue3+sigmoid(np.matmul(self.value_fn_params,oldAbsoluteState3))*(1-oldStateValue3)
        valueError=totalReward+(0.1*currentStateValue3)-oldStateValue3
        valueErrorBoard=totalBoardReward+(0.1*currentStateValue3)-oldStateValue3
        valueErrorMesag=totalMesagReward+(0.1*currentStateValue3)-oldStateValue3
        #self.value_fn_params=self.value_fn_params+stepSizeValue*valueError*value_derivative*oldAbsoluteState3
        self.value_fn_params=self.value_fn_params+stepSizeValue*valueError*value_derivative*oldAbsoluteState3
        
        ####This code calculates the new values for the board policy####
        stepSizeBoard=0.01
        standard_deviation_board=[math.exp(y) for y in np.matmul(self.board_policy_std_params,oldAbsoluteState3)]
        mean_board=self.board_policy_backprop_info[-2]
        ##
        p1=[1/x**2 for x in standard_deviation_board]
        p2=self.board_policy_backprop_info[-1]-(np.matmul(self.board_policy_params,oldAbsoluteState3)+self.board_policy_biases)
        #p2=[stepSizeBoard-x for x in mean_board]
        p3=np.multiply(p1,p2)
        self.board_policy_biases=self.board_policy_biases+[stepSizeBoard*x for x in p3]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p3=flatten(p3)
        oldAbsoluteState3=flatten(oldAbsoluteState3)
        p4=np.dot(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState3)))))
        #p4=np.matmul(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState3)))))
        gradient_mean_board=p4
        ##
        ##
        p1=p1 #this stays the same from the other part
        p2=[x**2 for x in p2]
        p3=np.multiply(p1,p2)
        p4=[x-1 for x in p3]
        self.board_policy_std_biases=self.board_policy_std_biases+[stepSizeBoard*x for x in p4]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p4=flatten(p4)
        oldAbsoluteState3=flatten(oldAbsoluteState3)
        p5=np.dot(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState3)))))
        #p5=np.matmul(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState3)))))
        gradient_std_board=p5
        ##
        #self.board_policy_params=self.board_policy_params+np.multiply((stepSizeBoard*self.I*valueError),gradient_mean_board)
        #self.board_policy_std_params=self.board_policy_std_params+np.multiply((stepSizeBoard*self.I*valueError),gradient_std_board)
        self.board_policy_params=self.board_policy_params+np.multiply((stepSizeBoard*self.I*valueErrorBoard),gradient_mean_board)
        self.board_policy_std_params=self.board_policy_std_params+np.multiply((stepSizeBoard*self.I*valueErrorBoard),gradient_std_board)
        
        ####This code calculates the new values for the message policy####
        stepSizeMessage=0.01
        standard_deviation_message=[math.exp(y) for y in np.matmul(self.msg_policy_std_params,oldAbsoluteState3)]
        mean_message=self.msg_policy_backprop_info[-2]
        ##
        p1=[1/x**2 for x in standard_deviation_message]
        p2=self.msg_policy_backprop_info[-1]-(np.matmul(self.msg_policy_params,oldAbsoluteState3)+self.msg_policy_biases)
        #p2=[stepSizeMessage-x for x in mean_message]
        p3=np.multiply(p1,p2)
        self.msg_policy_biases=self.msg_policy_biases+[stepSizeMessage*x for x in p3]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p3=flatten(p3)
        oldAbsoluteState3=flatten(oldAbsoluteState3)
        p4=np.dot(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState3)))))
        #p4=np.matmul(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState3)))))
        gradient_mean_message=p4
        ##
        ##
        p1=p1 #this stays the same from the other part
        p2=[x**2 for x in p2]
        p3=np.multiply(p1,p2)
        p4=[x-1 for x in p3]
        self.msg_policy_std_biases=self.msg_policy_std_biases+[stepSizeMessage*x for x in p4]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p4=flatten(p4)
        oldAbsoluteState3=flatten(oldAbsoluteState3)
        p5=np.dot(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState3)))))
        #p5=np.matmul(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState3)))))
        gradient_std_message=p5
        ##
        #self.msg_policy_params=self.msg_policy_params+np.multiply((stepSizeMessage*self.I*valueError),gradient_mean_message)
        #self.msg_policy_std_params=self.msg_policy_std_params+np.multiply((stepSizeMessage*self.I*valueError),gradient_std_message)
        self.msg_policy_params=self.msg_policy_params+np.multiply((stepSizeMessage*self.I*valueErrorMesag),gradient_mean_message)
        self.msg_policy_std_params=self.msg_policy_std_params+np.multiply((stepSizeMessage*self.I*valueErrorMesag),gradient_std_message)

        
        self.I=self.I*valueDiscount
        ####This code does gradient passing back to the FFN's####
        #d=ffn learning rate
        d=0.01
        valueError=valueError
        oldStateValue3=oldStateValue3
        ffn_out_one=self.absolute_state_history[-2][-1]#same as oldstate3
        ffn_out_zero=self.absolute_state_history[-2][-2]
        abs_state=self.absolute_state_history[-2][-3]
        backprop1=valueError*(oldStateValue3+sigmoid(np.matmul(self.value_fn_params,ffn_out_one)*(1-oldStateValue3)))*ffn_out_one
        olderror=np.transpose(valueError*(oldStateValue3+sigmoid(np.matmul(self.value_fn_params,ffn_out_one)*(1-oldStateValue3)))*self.value_fn_params)
        
        backprop2=np.matmul((olderror*np.array(list(zip(ffn_out_one+sigmoid(np.matmul(self.ffn_params[1],ffn_out_zero))*(1-ffn_out_one))))),np.transpose(np.array(list(zip(ffn_out_zero)))))
        olderror2=np.transpose(np.matmul(np.transpose((olderror*np.array(list(zip(ffn_out_one+sigmoid(np.matmul(self.ffn_params[1],ffn_out_zero))*(1-ffn_out_one)))))),self.ffn_params[1]))
        
        backprop3=np.matmul((olderror2*np.array(list(zip((ffn_out_zero+sigmoid(np.matmul(self.ffn_params[0],abs_state))*(1-ffn_out_zero)))))),np.transpose(np.array(list(zip(abs_state)))))
        olderror3=np.matmul(np.transpose(self.ffn_params[0]),(olderror2*np.array(list(zip((ffn_out_zero+sigmoid(np.matmul(self.ffn_params[0],abs_state))*(1-ffn_out_zero)))))))

        self.ffn_params[1]=self.ffn_params[1]+d*backprop2
        self.ffn_params[0]=self.ffn_params[0]+d*backprop3

        new_bp2=[d*x for x in (olderror*np.array(list(zip(ffn_out_one+sigmoid(np.matmul(self.ffn_params[1],ffn_out_zero))*(1-ffn_out_one)))))]
        new_bp3=[d*x for x in (olderror2*np.array(list(zip((ffn_out_zero+sigmoid(np.matmul(self.ffn_params[0],abs_state))*(1-ffn_out_zero))))))]

        for x in range(len(self.ffn_biases[1])):
            self.ffn_biases[1][x]=self.ffn_biases[1][x]+new_bp2[x]
        for x in range(len(self.ffn_biases[0])):
            self.ffn_biases[0][x]=self.ffn_biases[0][x]+new_bp3[x]

        ####This code does gradient passing back to LSTM's####
        #r=lstm learning rate
        r=0.01
        lstm_grads=[]
        lstm_grads.append(olderror3[0:300])
        lstm_grads.append(olderror3[300:600])
        lstm_grads.append(olderror3[600:900])
        lstm_grads.append(olderror3[900:1200])
        lstm_grads.append(olderror3[1200:1400])
        lstm_grads.append(olderror3[1400:1450])

        relations_old=[]
        for y in self.lstm_inputs[-2][-1]:
            relations_old.append(y[0])
        self.lstm_inputs[-2][-1]=relations_old
        for x in range(len(lstm_grads)):
            new_lstm_grads=[]
            for y in lstm_grads[x]:
                new_lstm_grads.append(y[0])
            
            oaError=np.multiply(new_lstm_grads[x],tanh(self.lstm_recursive_info[x][0][-2]))
            ctError=np.multiply(np.multiply(new_lstm_grads[x],self.lstm_backprop_state[x][3][-2]),[1-x**2 for x in tanh(self.lstm_recursive_info[x][0][-2])])
            iaError=np.multiply(ctError,self.lstm_backprop_state[x][2][-2])
            gaError=np.multiply(ctError,self.lstm_backprop_state[x][1][-2])
            faError=np.multiply(ctError,self.lstm_recursive_info[x][0][-3])
            
            gaError=np.multiply(gaError,[1-x**2 for x in np.arctanh(self.lstm_backprop_state[x][2][-2])])
            iaError=np.multiply(np.multiply(iaError,self.lstm_backprop_state[x][1][-2]),[1-x for x in self.lstm_backprop_state[x][1][-2]])
            faError=np.multiply(np.multiply(faError,self.lstm_backprop_state[x][0][-2]),[1-x for x in self.lstm_backprop_state[x][0][-2]])
            oaError=np.multiply(np.multiply(oaError,self.lstm_backprop_state[x][3][-2]),[1-x for x in self.lstm_backprop_state[x][3][-2]])

            concat_input=np.concatenate([self.lstm_inputs[-2][x],self.lstm_recursive_info[x][1][-3]])
            #this has an error on the last part with the relation values
            
            self.lstm_params[x][0]=self.lstm_params[x][0]+r*np.matmul(np.array(list(zip(faError))),np.transpose(np.array(list(zip(concat_input)))))
            self.lstm_params[x][1]=self.lstm_params[x][1]+r*np.matmul(np.array(list(zip(iaError))),np.transpose(np.array(list(zip(concat_input)))))
            self.lstm_params[x][2]=self.lstm_params[x][2]+r*np.matmul(np.array(list(zip(gaError))),np.transpose(np.array(list(zip(concat_input)))))
            self.lstm_params[x][3]=self.lstm_params[x][3]+r*np.matmul(np.array(list(zip(oaError))),np.transpose(np.array(list(zip(concat_input)))))

            self.lstm_biases[x][0]=np.transpose(self.lstm_biases[x][0]+r*faError)
            self.lstm_biases[x][1]=np.transpose(self.lstm_biases[x][1]+r*iaError)
            self.lstm_biases[x][2]=np.transpose(self.lstm_biases[x][2]+r*gaError)
            self.lstm_biases[x][3]=np.transpose(self.lstm_biases[x][3]+r*oaError)

        return relations, totalReward, boardReward, generalReward, totalSpeakingReward, totalListeningReward
        
class agentTwo():
    def __init__(self):
        #each lstm has 4 weights and 4 biases see https://www.kaggle.com/navjindervirdee/lstm-neural-network-from-scratch and http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        #lstm params has msg1-4 then board params in the order fa, ia, ga, oa with bias params
        #lstm backprop state has msg1-4 then board in the order fa, ia, ga, oa for back propogation
        #lstm recursive info has msg1-4 then board with the cell state then the last activation matrix
        #100 input length, 300 hidden length, so 400 total input length
        self.lstm_params=[[np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400))],[np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400))],[np.random.normal(0,0.01,(200,224)),np.random.normal(0,0.01,(200,224)),np.random.normal(0,0.01,(200,224)),np.random.normal(0,0.01,(200,224))],[np.random.normal(0,0.01,(50,51)),np.random.normal(0,0.01,(50,51)),np.random.normal(0,0.01,(50,51)),np.random.normal(0,0.01,(50,51))]]
        self.lstm_backprop_state=[[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
        self.lstm_recursive_info=[[[np.ones(300)],[np.ones(300)]],[[np.ones(300)],[np.ones(300)]],[[np.ones(200)],[np.ones(200)]],[[np.ones(50)],[np.ones(50)]]]
        self.lstm_biases=[[np.zeros(300),np.zeros(300),np.zeros(300),np.zeros(300)],[np.zeros(300),np.zeros(300),np.zeros(300),np.zeros(300)],[np.zeros(200),np.zeros(200),np.zeros(200),np.zeros(200)],[np.zeros(50),np.zeros(50),np.zeros(50),np.zeros(50)]]

        #ffn is gonna have 2 layers each with relu that cut the length down. 850->425->200
        self.ffn_params=[np.random.normal(0,0.01,(425,850)),np.random.normal(0,0.01,(200,425))]
        self.ffn_biases=[np.zeros(425),np.zeros(200)]
        self.ffn_outputs=[[],[]]

        #im using linear value function approximation
        self.value_fn_params=np.random.normal(0,0.01,(1,200))

        self.board_policy_params=[np.random.normal(0,0.01,(7,200))]
        self.board_policy_std_params=[np.random.normal(0,0.01,(7,200))]
        self.board_policy_biases=np.zeros(7)
        self.board_policy_std_biases=np.zeros(7)
        self.board_policy_backprop_info=[np.zeros(8)]
        
        self.msg_policy_params=[np.random.normal(0,0.01,(100,200))]
        self.msg_policy_std_params=[np.random.normal(0,0.01,(100,200))]
        self.msg_policy_biases=np.zeros(100)
        self.msg_policy_std_biases=np.zeros(100)
        self.msg_policy_backprop_info=[np.zeros(100)]

        #the following three lists hold first every absolute state that the agent takes an action from, second every message the agent generates, and third every action the agent takes. They go by time step
        self.absolute_state_history=[]
        self.message_history_self=[np.zeros(100)]
        self.action_history_self=[np.zeros(8)]

        #used for message reward
        self.msgStateHistory=[]
        
        #this holds the messages of the other agents
        self.external_message_history=[]

        self.I=1

        self.boardReward_history=[]
        self.generalReward_history=[]
        self.totalSpeakingReward_history=[]
        self.totalListeningReward_history=[]

        self.lstm_inputs=[]
        
    def stateConcatTwo(self, fullState):
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
        relationVal=[fullState[4][1]]
        x=0
        for y in self.external_message_history:
            if (y==messageTwo).all():
                x=1
        if x==0:
            self.external_message_history.append(messageTwo)
        lstm_inputs=[messageTwo,messageFour,boardState,relationVal]
        self.lstm_inputs.append(lstm_inputs)
        for x in range(len(lstm_inputs)):
            concat_input=np.concatenate((lstm_inputs[x],self.lstm_recursive_info[x][1][-1]))

            fa=sigmoid(np.matmul(self.lstm_params[x][0],concat_input)+self.lstm_biases[x][0])

            ia=sigmoid(np.matmul(self.lstm_params[x][1],concat_input)+self.lstm_biases[x][1])

            ga=tanh(np.matmul(self.lstm_params[x][2],concat_input)+self.lstm_biases[x][2])

            oa=sigmoid(np.matmul(self.lstm_params[x][3],concat_input)+self.lstm_biases[x][3])

            ct=np.multiply(fa,self.lstm_recursive_info[x][0][-1])+np.multiply(ia,ga)

            am=np.multiply(oa, tanh(ct))
            
            self.lstm_backprop_state[x][0].append(fa)
            self.lstm_backprop_state[x][1].append(ia)
            self.lstm_backprop_state[x][2].append(ga)
            self.lstm_backprop_state[x][3].append(oa)
            self.lstm_recursive_info[x][0].append(ct)
            self.lstm_recursive_info[x][1].append(am)

        absolute_state=np.concatenate((self.lstm_recursive_info[0][1][-1],self.lstm_recursive_info[1][1][-1],self.lstm_recursive_info[2][1][-1],self.lstm_recursive_info[3][1][-1]))
        self.ffn_outputs[0]=relu(np.matmul(self.ffn_params[0],absolute_state)+self.ffn_biases[0])
        self.ffn_outputs[1]=relu(np.matmul(self.ffn_params[1],self.ffn_outputs[0])+self.ffn_biases[1])
        #ffn_2 is the networks perception of the state distilled from memory and feed forward layers; len 200
        self.absolute_state_history.append([absolute_state,self.ffn_outputs[0],self.ffn_outputs[1]])
        return self.ffn_outputs[1]

    def boardAction(self):
        nn8=np.matmul(self.board_policy_params,self.ffn_outputs[1])+self.board_policy_biases
        nn8=np.insert(nn8,0,2)
        alternateAction=[2,(np.random.random()*10)-10,(np.random.random()*8),(np.random.random()*8),(np.random.random()*8),(np.random.random()*8),(np.random.random()*8),(np.random.random()*8)]
        orig_action=[]
        action=[]
        if np.random.randint(1,11)<6:
            orig_action=alternateAction
            action=alternateAction
        else:
            orig_action=nn8
            action=nn8
        action[1]=np.round((5/(1+np.exp(-action[1]-5))))
        for x in range(len(action[2:])):
            action[x+2]=np.round((4/(1+np.exp(-(action[x+2]-4))))-2)
        boardAction=[action[0],action[1],action[2:]]
        self.board_policy_backprop_info.append(orig_action)
        self.action_history_self.append(action)
        return boardAction
        
    def messageAction(self):
        nn10=np.matmul(self.msg_policy_params,self.ffn_outputs[1])+self.msg_policy_biases
        message4=nn10
        messageAction=message4[0]
        self.msg_policy_backprop_info.append(nn10)
        self.message_history_self.append(nn10)
        self.msgStateHistory.append(self.ffn_outputs[1])
        return messageAction
    
    def stateValue(self, absoluteState2):
        currentStateValue2=np.matmul(self.value_fn_params,absoluteState2)
        return currentStateValue2

    def updateParameters(self,oldAbsoluteState2,absoluteState2,reward):
        ####This code regularizes the reward:####
        self.boardReward_history.append(boardReward)
        self.generalReward_history.append(generalReward)
        self.totalSpeakingReward_history.append(totalSpeakingReward)
        self.totalListeningReward_history.append(totalListeningReward)

        ####This code amalgamates the reward into one value:####
        totalReward=boardReward+generalReward+totalSpeakingReward+totalListeningReward
        totalBoardReward=boardReward+generalReward
        totalMesagReward=generalReward+totalSpeakingReward+totalListeningReward
        ##This code calculates the value of the current state
        currentStateValue2=swish(np.matmul(self.value_fn_params,absoluteState2))
        oldStateValue2=swish(np.matmul(self.value_fn_params,oldAbsoluteState2))

        
        ####This code calculates the value function error, which is important in getting the error of the other states. It also updates the value function####
        valueDiscount=0.1
        stepSizeValue=0.01
        #y'=y+sigmoid(x)*(1-y)...this is multiplied by the old state because to do the backprop thing u have to back propogate all the way from the value to the input
        value_derivative=oldStateValue2+sigmoid(np.matmul(self.value_fn_params,oldAbsoluteState2))*(1-oldStateValue2)
        valueError=totalReward+(0.1*currentStateValue2)-oldStateValue2
        valueErrorBoard=totalBoardReward+(0.1*currentStateValue2)-oldStateValue2
        valueErrorMesag=totalMesagReward+(0.1*currentStateValue2)-oldStateValue2
        #self.value_fn_params=self.value_fn_params+stepSizeValue*valueError*value_derivative*oldAbsoluteState2
        self.value_fn_params=self.value_fn_params+stepSizeValue*valueError*value_derivative
        
        ####This code calculates the new values for the board policy####
        stepSizeBoard=0.01
        standard_deviation_board=[math.exp(y) for y in np.transpose(np.matmul(self.board_policy_std_params,oldAbsoluteState2))]
        mean_board=self.board_policy_backprop_info[-2][1:]
        ##
        p1=[1/x**2 for x in standard_deviation_board]
        p2=self.board_policy_backprop_info[-1][1:]-(np.matmul(self.board_policy_params,oldAbsoluteState2)+self.board_policy_biases)
        #p2=[stepSizeBoard-x for x in mean_board]
        p3=np.multiply(p1,p2)
        self.board_policy_biases=self.board_policy_biases+[stepSizeBoard*x for x in p3]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p3=flatten(p3)
        oldAbsoluteState2=flatten(oldAbsoluteState2)
        p4=np.dot(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState2)))))
        #p4=np.matmul(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState2)))))
        gradient_mean_board=p4
        ##
        ##
        p1=p1 #this stays the same from the other part
        p2=[x**2 for x in p2]
        p3=np.multiply(p1,p2)
        p4=[x-1 for x in p3]
        self.board_policy_std_biases=self.board_policy_std_biases+[stepSizeBoard*x for x in p4]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p4=flatten(p4)
        oldAbsoluteState2=flatten(oldAbsoluteState2)
        p5=np.dot(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState2)))))
        #p5=np.matmul(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState2)))))
        gradient_std_board=p5
        ##
        #self.board_policy_params=self.board_policy_params+np.multiply((stepSizeBoard*self.I*valueError),gradient_mean_board)
        #self.board_policy_std_params=self.board_policy_std_params+np.multiply((stepSizeBoard*self.I*valueError),gradient_std_board)
        self.board_policy_params=self.board_policy_params+np.multiply((stepSizeBoard*self.I*valueErrorBoard),gradient_mean_board)
        self.board_policy_std_params=self.board_policy_std_params+np.multiply((stepSizeBoard*self.I*valueErrorBoard),gradient_std_board)
        
        ####This code calculates the new values for the message policy####
        stepSizeMessage=0.01
        standard_deviation_message=[math.exp(y) for y in np.transpose(np.matmul(self.msg_policy_std_params,oldAbsoluteState2))]
        mean_message=self.msg_policy_backprop_info[-2]
        ##
        p1=[1/x**2 for x in standard_deviation_message]
        p2=self.msg_policy_backprop_info[-1]-(np.matmul(self.msg_policy_params,oldAbsoluteState2)+self.msg_policy_biases)
        #p2=[stepSizeMessage-x for x in mean_message]
        p3=np.multiply(p1,p2)
        self.msg_policy_biases=self.msg_policy_biases+[stepSizeMessage*x for x in p3]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p3=flatten(p3)
        oldAbsoluteState2=flatten(oldAbsoluteState2)
        p4=np.dot(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState2)))))
        #p4=np.matmul(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState2)))))
        gradient_mean_message=p4
        ##
        ##
        p1=p1 #this stays the same from the other part
        p2=[x**2 for x in p2]
        p3=np.multiply(p1,p2)
        p4=[x-1 for x in p3]
        self.msg_policy_std_biases=self.msg_policy_std_biases+[stepSizeMessage*x for x in p4]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p4=flatten(p4)
        oldAbsoluteState2=flatten(oldAbsoluteState2)
        p5=np.dot(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState2)))))
        #p5=np.matmul(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState2)))))
        gradient_std_message=p5
        ##
        #self.msg_policy_params=self.msg_policy_params+np.multiply((stepSizeMessage*self.I*valueError),gradient_mean_message)
        #self.msg_policy_std_params=self.msg_policy_std_params+np.multiply((stepSizeMessage*self.I*valueError),gradient_std_message)
        self.msg_policy_params=self.msg_policy_params+np.multiply((stepSizeMessage*self.I*valueErrorMesag),gradient_mean_message)
        self.msg_policy_std_params=self.msg_policy_std_params+np.multiply((stepSizeMessage*self.I*valueErrorMesag),gradient_std_message)
        
        self.I=self.I*valueDiscount
        ####This code does gradient passing back to the FFN's####
        #d=ffn learning rate
        d=0.01
        valueError=valueError
        oldStateValue3=oldStateValue2
        ffn_out_one=self.absolute_state_history[-2][-1]#same as oldstate3
        ffn_out_zero=self.absolute_state_history[-2][-2]
        abs_state=self.absolute_state_history[-2][-3]
        backprop1=valueError*(oldStateValue2+sigmoid(np.matmul(self.value_fn_params,ffn_out_one)*(1-oldStateValue2)))*ffn_out_one
        olderror=np.transpose(valueError*(oldStateValue2+sigmoid(np.matmul(self.value_fn_params,ffn_out_one)*(1-oldStateValue2)))*self.value_fn_params)
        
        backprop2=np.matmul((olderror*np.array(list(zip(ffn_out_one+sigmoid(np.matmul(self.ffn_params[1],ffn_out_zero))*(1-ffn_out_one))))),np.transpose(np.array(list(zip(ffn_out_zero)))))
        olderror2=np.transpose(np.matmul(np.transpose((olderror*np.array(list(zip(ffn_out_one+sigmoid(np.matmul(self.ffn_params[1],ffn_out_zero))*(1-ffn_out_one)))))),self.ffn_params[1]))
        
        backprop3=np.matmul((olderror2*np.array(list(zip((ffn_out_zero+sigmoid(np.matmul(self.ffn_params[0],abs_state))*(1-ffn_out_zero)))))),np.transpose(np.array(list(zip(abs_state)))))
        olderror3=np.matmul(np.transpose(self.ffn_params[0]),(olderror2*np.array(list(zip((ffn_out_zero+sigmoid(np.matmul(self.ffn_params[0],abs_state))*(1-ffn_out_zero)))))))

        self.ffn_params[1]=self.ffn_params[1]+d*backprop2
        self.ffn_params[0]=self.ffn_params[0]+d*backprop3

        new_bp2=[d*x for x in (olderror*np.array(list(zip(ffn_out_one+sigmoid(np.matmul(self.ffn_params[1],ffn_out_zero))*(1-ffn_out_one)))))]
        new_bp3=[d*x for x in (olderror2*np.array(list(zip((ffn_out_zero+sigmoid(np.matmul(self.ffn_params[0],abs_state))*(1-ffn_out_zero))))))]

        for x in range(len(self.ffn_biases[1])):
            self.ffn_biases[1][x]=self.ffn_biases[1][x]+new_bp2[x]
        for x in range(len(self.ffn_biases[0])):
            self.ffn_biases[0][x]=self.ffn_biases[0][x]+new_bp3[x]

        ####This code does gradient passing back to LSTM's####
        #r=lstm learning rate
        r=0.01
        lstm_grads=[]
        lstm_grads.append(olderror3[0:300])
        lstm_grads.append(olderror3[300:600])
        lstm_grads.append(olderror3[600:800])
        lstm_grads.append(olderror3[800:850])

        relations_old=[]
        for y in self.lstm_inputs[-2][-1]:
            relations_old.append(y)
        self.lstm_inputs[-2][-1]=relations_old
        for x in range(len(lstm_grads)):
            new_lstm_grads=[]
            for y in lstm_grads[x]:
                new_lstm_grads.append(y[0])
            
            oaError=np.multiply(new_lstm_grads[x],tanh(self.lstm_recursive_info[x][0][-2]))
            ctError=np.multiply(np.multiply(new_lstm_grads[x],self.lstm_backprop_state[x][3][-2]),[1-x**2 for x in tanh(self.lstm_recursive_info[x][0][-2])])
            iaError=np.multiply(ctError,self.lstm_backprop_state[x][2][-2])
            gaError=np.multiply(ctError,self.lstm_backprop_state[x][1][-2])
            faError=np.multiply(ctError,self.lstm_recursive_info[x][0][-3])
            
            gaError=np.multiply(gaError,[1-x**2 for x in np.arctanh(self.lstm_backprop_state[x][2][-2])])
            iaError=np.multiply(np.multiply(iaError,self.lstm_backprop_state[x][1][-2]),[1-x for x in self.lstm_backprop_state[x][1][-2]])
            faError=np.multiply(np.multiply(faError,self.lstm_backprop_state[x][0][-2]),[1-x for x in self.lstm_backprop_state[x][0][-2]])
            oaError=np.multiply(np.multiply(oaError,self.lstm_backprop_state[x][3][-2]),[1-x for x in self.lstm_backprop_state[x][3][-2]])

            concat_input=np.concatenate([self.lstm_inputs[-2][x],self.lstm_recursive_info[x][1][-3]])
            #this has an error on the last part with the relation values
            
            self.lstm_params[x][0]=self.lstm_params[x][0]+r*np.matmul(np.array(list(zip(faError))),np.transpose(np.array(list(zip(concat_input)))))
            self.lstm_params[x][1]=self.lstm_params[x][1]+r*np.matmul(np.array(list(zip(iaError))),np.transpose(np.array(list(zip(concat_input)))))
            self.lstm_params[x][2]=self.lstm_params[x][2]+r*np.matmul(np.array(list(zip(gaError))),np.transpose(np.array(list(zip(concat_input)))))
            self.lstm_params[x][3]=self.lstm_params[x][3]+r*np.matmul(np.array(list(zip(oaError))),np.transpose(np.array(list(zip(concat_input)))))

            self.lstm_biases[x][0]=np.transpose(self.lstm_biases[x][0]+r*faError)
            self.lstm_biases[x][1]=np.transpose(self.lstm_biases[x][1]+r*iaError)
            self.lstm_biases[x][2]=np.transpose(self.lstm_biases[x][2]+r*gaError)
            self.lstm_biases[x][3]=np.transpose(self.lstm_biases[x][3]+r*oaError)

        return totalReward, boardReward, generalReward, totalSpeakingReward, totalListeningReward

        
            
class agentOne():
    def __init__(self):
        #each lstm has 4 weights and 4 biases see https://www.kaggle.com/navjindervirdee/lstm-neural-network-from-scratch and http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        #lstm params has msg1-4 then board params in the order fa, ia, ga, oa with bias params
        #lstm backprop state has msg1-4 then board in the order fa, ia, ga, oa for back propogation
        #lstm recursive info has msg1-4 then board with the cell state then the last activation matrix
        #100 input length, 300 hidden length, so 400 total input length
        self.lstm_params=[[np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400))],[np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400)),np.random.normal(0,0.01,(300,400))],[np.random.normal(0,0.01,(200,224)),np.random.normal(0,0.01,(200,224)),np.random.normal(0,0.01,(200,224)),np.random.normal(0,0.01,(200,224))],[np.random.normal(0,0.01,(50,51)),np.random.normal(0,0.01,(50,51)),np.random.normal(0,0.01,(50,51)),np.random.normal(0,0.01,(50,51))]]
        self.lstm_backprop_state=[[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
        self.lstm_recursive_info=[[[np.ones(300)],[np.ones(300)]],[[np.ones(300)],[np.ones(300)]],[[np.ones(200)],[np.ones(200)]],[[np.ones(50)],[np.ones(50)]]]
        self.lstm_biases=[[np.zeros(300),np.zeros(300),np.zeros(300),np.zeros(300)],[np.zeros(300),np.zeros(300),np.zeros(300),np.zeros(300)],[np.zeros(200),np.zeros(200),np.zeros(200),np.zeros(200)],[np.zeros(50),np.zeros(50),np.zeros(50),np.zeros(50)]]

        #ffn is gonna have 2 layers each with relu that cut the length down. 850->425->200
        self.ffn_params=[np.random.normal(0,0.01,(425,850)),np.random.normal(0,0.01,(200,425))]
        self.ffn_biases=[np.zeros(425),np.zeros(200)]
        self.ffn_outputs=[[],[]]
        
        #im using linear value function approximation
        self.value_fn_params=np.random.normal(0,0.01,(1,200))
        
        self.board_policy_params=[np.random.normal(0,0.01,(7,200))]
        self.board_policy_std_params=[np.random.normal(0,0.01,(7,200))]
        self.board_policy_biases=np.zeros(7)
        self.board_policy_std_biases=np.zeros(7)
        self.board_policy_backprop_info=[np.zeros(8)]
        
        self.msg_policy_params=[np.random.normal(0,0.01,(100,200))]
        self.msg_policy_std_params=[np.random.normal(0,0.01,(100,200))]
        self.msg_policy_biases=np.zeros(100)
        self.msg_policy_std_biases=np.zeros(100)
        self.msg_policy_backprop_info=[np.zeros(100)]

        #the following three lists hold first every absolute state that the agent takes an action from, second every message the agent generates, and third every action the agent takes. They go by time step
        self.absolute_state_history=[]
        self.message_history_self=[np.zeros(100)]
        self.action_history_self=[np.zeros(8)]

        #used for message reward
        self.msgStateHistory=[]

        #this holds the messages of the other agents
        self.external_message_history=[]

        self.I=1

        self.boardReward_history=[]
        self.generalReward_history=[]
        self.totalSpeakingReward_history=[]
        self.totalListeningReward_history=[]

        self.lstm_inputs=[]

        
    def stateConcatOne(self, fullState):
        reward=fullState[1]
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
        relationVal=[fullState[4][0]]
        x=0
        for y in self.external_message_history:
            if (y==messageOne).all():
                x=1
        if x==0:
            self.external_message_history.append(messageOne)
        lstm_inputs=[messageOne,messageThree,boardState,relationVal]
        self.lstm_inputs.append(lstm_inputs)
        for x in range(len(lstm_inputs)):
            concat_input=np.concatenate((lstm_inputs[x],self.lstm_recursive_info[x][1][-1]))

            fa=sigmoid(np.matmul(self.lstm_params[x][0],concat_input)+self.lstm_biases[x][0])

            ia=sigmoid(np.matmul(self.lstm_params[x][1],concat_input)+self.lstm_biases[x][1])

            ga=tanh(np.matmul(self.lstm_params[x][2],concat_input)+self.lstm_biases[x][2])

            oa=sigmoid(np.matmul(self.lstm_params[x][3],concat_input)+self.lstm_biases[x][3])

            ct=np.multiply(fa,self.lstm_recursive_info[x][0][-1])+np.multiply(ia,ga)

            am=np.multiply(oa, tanh(ct))

            self.lstm_backprop_state[x][0].append(fa)
            self.lstm_backprop_state[x][1].append(ia)
            self.lstm_backprop_state[x][2].append(ga)
            self.lstm_backprop_state[x][3].append(oa)
            self.lstm_recursive_info[x][0].append(ct)
            self.lstm_recursive_info[x][1].append(am)

        absolute_state=np.concatenate((self.lstm_recursive_info[0][1][-1],self.lstm_recursive_info[1][1][-1],self.lstm_recursive_info[2][1][-1],self.lstm_recursive_info[3][1][-1]))
        self.ffn_outputs[0]=relu(np.matmul(self.ffn_params[0],absolute_state)+self.ffn_biases[0])
        self.ffn_outputs[1]=relu(np.matmul(self.ffn_params[1],self.ffn_outputs[0])+self.ffn_biases[1])
        #ffn_2 is the networks perception of the state distilled from memory and feed forward layers; len 200
        self.absolute_state_history.append([absolute_state,self.ffn_outputs[0],self.ffn_outputs[1]])
        return self.ffn_outputs[1]

    def boardAction(self):
        nn8=np.matmul(self.board_policy_params,self.ffn_outputs[1])+self.board_policy_biases
        nn8=np.insert(nn8,0,1)
        alternateAction=[1,(np.random.random()*10)-10,(np.random.random()*8),(np.random.random()*8),(np.random.random()*8),(np.random.random()*8),(np.random.random()*8),(np.random.random()*8)]
        orig_action=[]
        action=[]
        if np.random.randint(1,11)<6:
            orig_action=alternateAction
            action=alternateAction
        else:
            orig_action=nn8
            action=nn8
        action[1]=np.round((5/(1+np.exp(-action[1]-5))))
        for x in range(len(action[2:])):
            action[x+2]=np.round((4/(1+np.exp(-(action[x+2]-4))))-2)
        boardAction=[action[0],action[1],action[2:]]
        self.board_policy_backprop_info.append(orig_action)
        self.action_history_self.append(action)
        return boardAction
        
    def messageAction(self):
        nn10=np.matmul(self.msg_policy_params,self.ffn_outputs[1])+self.msg_policy_biases
        message3=nn10
        messageAction=message3[0]
        self.msg_policy_backprop_info.append(nn10)
        self.message_history_self.append(nn10)
        self.msgStateHistory.append(self.ffn_outputs[1])
        return messageAction

    def stateValue(self, absoluteState1):
        currentStateValue1=np.matmul(self.value_fn_params,absoluteState1)
        return currentStateValue1

    def updateParameters(self,oldAbsoluteState1,absoluteState1,reward):
        ####This code regularizes the reward:####
        self.boardReward_history.append(boardReward)
        self.generalReward_history.append(generalReward)
        self.totalSpeakingReward_history.append(totalSpeakingReward)
        self.totalListeningReward_history.append(totalListeningReward)

        ####This code amalgamates the reward into one value:####
        totalReward=boardReward+generalReward+totalSpeakingReward+totalListeningReward
        totalBoardReward=boardReward+generalReward
        totalMesagReward=generalReward+totalSpeakingReward+totalListeningReward
        ##This code calculates the value of the current state
        currentStateValue1=swish(np.matmul(self.value_fn_params,absoluteState1))
        oldStateValue1=swish(np.matmul(self.value_fn_params,oldAbsoluteState1))
        
        ####This code calculates the value function error, which is important in getting the error of the other states. It also updates the value function####
        valueDiscount=0.1
        stepSizeValue=0.01
        #y'=y+sigmoid(x)*(1-y)...this is multiplied by the old state because to do the backprop thing u have to back propogate all the way from the value to the input
        value_derivative=oldStateValue1+sigmoid(np.matmul(self.value_fn_params,oldAbsoluteState1))*(1-oldStateValue1)
        valueError=totalReward+(0.1*currentStateValue1)-oldStateValue1
        valueErrorBoard=totalBoardReward+(0.1*currentStateValue1)-oldStateValue1
        valueErrorMesag=totalMesagReward+(0.1*currentStateValue1)-oldStateValue1
        #self.value_fn_params=self.value_fn_params+stepSizeValue*valueError*value_derivative*oldAbsoluteState1
        self.value_fn_params=self.value_fn_params+stepSizeValue*valueError*value_derivative
        
        ####This code calculates the new values for the board policy####
        stepSizeBoard=0.01
        standard_deviation_board=[math.exp(y) for y in np.transpose(np.matmul(self.board_policy_std_params,oldAbsoluteState1))]
        mean_board=self.board_policy_backprop_info[-2][1:]
        ##
        p1=[1/x**2 for x in standard_deviation_board]
        p2=self.board_policy_backprop_info[-1][1:]-(np.matmul(self.board_policy_params,oldAbsoluteState1)+self.board_policy_biases)
        #p2=[stepSizeBoard-x for x in mean_board]
        p3=np.multiply(p1,p2)
        self.board_policy_biases=self.board_policy_biases+[stepSizeBoard*x for x in p3]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p3=flatten(p3)
        oldAbsoluteState1=flatten(oldAbsoluteState1)
        p4=np.dot(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState1)))))
        #p4=np.matmul(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState1)))))
        gradient_mean_board=p4
        ##
        ##
        p1=p1 #this stays the same from the other part
        p2=[x**2 for x in p2]
        p3=np.multiply(p1,p2)
        p4=[x-1 for x in p3]
        self.board_policy_std_biases=self.board_policy_std_biases+[stepSizeBoard*x for x in p4]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p4=flatten(p4)
        oldAbsoluteState1=flatten(oldAbsoluteState1)
        p5=np.dot(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState1)))))
        #p5=np.matmul(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState1)))))
        gradient_std_board=p5
        ##
        #self.board_policy_params=self.board_policy_params+np.multiply((stepSizeBoard*self.I*valueError),gradient_mean_board)
        #self.board_policy_std_params=self.board_policy_std_params+np.multiply((stepSizeBoard*self.I*valueError),gradient_std_board)
        self.board_policy_params=self.board_policy_params+np.multiply((stepSizeBoard*self.I*valueErrorBoard),gradient_mean_board)
        self.board_policy_std_params=self.board_policy_std_params+np.multiply((stepSizeBoard*self.I*valueErrorBoard),gradient_std_board)
        
        ####This code calculates the new values for the message policy####
        stepSizeMessage=0.01
        standard_deviation_message=[math.exp(y) for y in np.transpose(np.matmul(self.msg_policy_std_params,oldAbsoluteState1))]
        mean_message=self.msg_policy_backprop_info[-2]
        ##
        p1=[1/x**2 for x in standard_deviation_message]
        p2=self.msg_policy_backprop_info[-1]-(np.matmul(self.msg_policy_params,oldAbsoluteState1)+self.msg_policy_biases)
        #p2=[stepSizeMessage-x for x in mean_message]
        p3=np.multiply(p1,p2)
        self.msg_policy_biases=self.msg_policy_biases+[stepSizeMessage*x for x in p3]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p3=flatten(p3)
        oldAbsoluteState1=flatten(oldAbsoluteState1)
        p4=np.dot(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState1)))))
        #p4=np.matmul(np.array(list(zip(p3))),np.transpose(np.array(list(zip(oldAbsoluteState1)))))
        gradient_mean_message=p4
        ##
        ##
        p1=p1 #this stays the same from the other part
        p2=[x**2 for x in p2]
        p3=np.multiply(p1,p2)
        p4=[x-1 for x in p3]
        self.msg_policy_std_biases=self.msg_policy_std_biases+[stepSizeMessage*x for x in p4]
        #
        #
        #
        #THERE WAS A CHANGE HERE
        #
        #
        #
        p4=flatten(p4)
        oldAbsoluteState1=flatten(oldAbsoluteState1)
        p5=np.dot(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState1)))))
        #p5=np.matmul(np.array(list(zip(p4))),np.transpose(np.array(list(zip(oldAbsoluteState1)))))
        gradient_std_message=p5
        ##
        #self.msg_policy_params=self.msg_policy_params+np.multiply((stepSizeMessage*self.I*valueError),gradient_mean_message)
        #self.msg_policy_std_params=self.msg_policy_std_params+np.multiply((stepSizeMessage*self.I*valueError),gradient_std_message)
        self.msg_policy_params=self.msg_policy_params+np.multiply((stepSizeMessage*self.I*valueErrorMesag),gradient_mean_message)
        self.msg_policy_std_params=self.msg_policy_std_params+np.multiply((stepSizeMessage*self.I*valueErrorMesag),gradient_std_message)
        
        self.I=self.I*valueDiscount
        ####This code does gradient passing back to the FFN's####
        #d=ffn learning rate
        d=0.01
        valueError=valueError
        oldStateValue1=oldStateValue1
        ffn_out_one=self.absolute_state_history[-2][-1]#same as oldstate3
        ffn_out_zero=self.absolute_state_history[-2][-2]
        abs_state=self.absolute_state_history[-2][-3]
        backprop1=valueError*(oldStateValue1+sigmoid(np.matmul(self.value_fn_params,ffn_out_one)*(1-oldStateValue1)))*ffn_out_one
        olderror=np.transpose(valueError*(oldStateValue1+sigmoid(np.matmul(self.value_fn_params,ffn_out_one)*(1-oldStateValue1)))*self.value_fn_params)
        
        backprop2=np.matmul((olderror*np.array(list(zip(ffn_out_one+sigmoid(np.matmul(self.ffn_params[1],ffn_out_zero))*(1-ffn_out_one))))),np.transpose(np.array(list(zip(ffn_out_zero)))))
        olderror2=np.transpose(np.matmul(np.transpose((olderror*np.array(list(zip(ffn_out_one+sigmoid(np.matmul(self.ffn_params[1],ffn_out_zero))*(1-ffn_out_one)))))),self.ffn_params[1]))
        
        backprop3=np.matmul((olderror2*np.array(list(zip((ffn_out_zero+sigmoid(np.matmul(self.ffn_params[0],abs_state))*(1-ffn_out_zero)))))),np.transpose(np.array(list(zip(abs_state)))))
        olderror3=np.matmul(np.transpose(self.ffn_params[0]),(olderror2*np.array(list(zip((ffn_out_zero+sigmoid(np.matmul(self.ffn_params[0],abs_state))*(1-ffn_out_zero)))))))

        self.ffn_params[1]=self.ffn_params[1]+d*backprop2
        self.ffn_params[0]=self.ffn_params[0]+d*backprop3

        new_bp2=[d*x for x in (olderror*np.array(list(zip(ffn_out_one+sigmoid(np.matmul(self.ffn_params[1],ffn_out_zero))*(1-ffn_out_one)))))]
        new_bp3=[d*x for x in (olderror2*np.array(list(zip((ffn_out_zero+sigmoid(np.matmul(self.ffn_params[0],abs_state))*(1-ffn_out_zero))))))]

        for x in range(len(self.ffn_biases[1])):
            self.ffn_biases[1][x]=self.ffn_biases[1][x]+new_bp2[x]
        for x in range(len(self.ffn_biases[0])):
            self.ffn_biases[0][x]=self.ffn_biases[0][x]+new_bp3[x]

        ####This code does gradient passing back to LSTM's####
        #r=lstm learning rate
        r=0.01
        lstm_grads=[]
        lstm_grads.append(olderror3[0:300])
        lstm_grads.append(olderror3[300:600])
        lstm_grads.append(olderror3[600:800])
        lstm_grads.append(olderror3[800:850])

        relations_old=[]
        for y in self.lstm_inputs[-2][-1]:
            relations_old.append(y)
        self.lstm_inputs[-2][-1]=relations_old
        for x in range(len(lstm_grads)):
            new_lstm_grads=[]
            for y in lstm_grads[x]:
                new_lstm_grads.append(y[0])
            
            oaError=np.multiply(new_lstm_grads[x],tanh(self.lstm_recursive_info[x][0][-2]))
            ctError=np.multiply(np.multiply(new_lstm_grads[x],self.lstm_backprop_state[x][3][-2]),[1-x**2 for x in tanh(self.lstm_recursive_info[x][0][-2])])
            iaError=np.multiply(ctError,self.lstm_backprop_state[x][2][-2])
            gaError=np.multiply(ctError,self.lstm_backprop_state[x][1][-2])
            faError=np.multiply(ctError,self.lstm_recursive_info[x][0][-3])
            
            gaError=np.multiply(gaError,[1-x**2 for x in np.arctanh(self.lstm_backprop_state[x][2][-2])])
            iaError=np.multiply(np.multiply(iaError,self.lstm_backprop_state[x][1][-2]),[1-x for x in self.lstm_backprop_state[x][1][-2]])
            faError=np.multiply(np.multiply(faError,self.lstm_backprop_state[x][0][-2]),[1-x for x in self.lstm_backprop_state[x][0][-2]])
            oaError=np.multiply(np.multiply(oaError,self.lstm_backprop_state[x][3][-2]),[1-x for x in self.lstm_backprop_state[x][3][-2]])

            concat_input=np.concatenate([self.lstm_inputs[-2][x],self.lstm_recursive_info[x][1][-3]])
            #this has an error on the last part with the relation values
            
            self.lstm_params[x][0]=self.lstm_params[x][0]+r*np.matmul(np.array(list(zip(faError))),np.transpose(np.array(list(zip(concat_input)))))
            self.lstm_params[x][1]=self.lstm_params[x][1]+r*np.matmul(np.array(list(zip(iaError))),np.transpose(np.array(list(zip(concat_input)))))
            self.lstm_params[x][2]=self.lstm_params[x][2]+r*np.matmul(np.array(list(zip(gaError))),np.transpose(np.array(list(zip(concat_input)))))
            self.lstm_params[x][3]=self.lstm_params[x][3]+r*np.matmul(np.array(list(zip(oaError))),np.transpose(np.array(list(zip(concat_input)))))

            self.lstm_biases[x][0]=np.transpose(self.lstm_biases[x][0]+r*faError)
            self.lstm_biases[x][1]=np.transpose(self.lstm_biases[x][1]+r*iaError)
            self.lstm_biases[x][2]=np.transpose(self.lstm_biases[x][2]+r*gaError)
            self.lstm_biases[x][3]=np.transpose(self.lstm_biases[x][3]+r*oaError)

        return totalReward, boardReward, generalReward, totalSpeakingReward, totalListeningReward


#output function to track
plot.ion()
plot.style.use('ggplot')
fig=plot.figure(figsize=(7,7))
episode=1
turn=1
agent1avg=0
agent2avg=0
agent3avg=0
episode_data=[]

def turn_output(agent1message,agent2message,agent3message,agent1action,agent2action,agent3action,total_reward_1,total_reward_2,total_reward_3,component1,component2,component3):
    global episode_data
    episode_data.append([agent1message,agent2message,agent3message,agent1action,agent2action,agent3action,total_reward_1,total_reward_2,total_reward_3[1:-2],component1,component2,component3])
    print(len(episode_data))

def episode_csv():
    global episode_data, episode
    with open('episode_'+str(episode)+'.csv','w',newline='') as csvfile:
        writer=csv.writer(csvfile,dialect='excel')
        for x in episode_data:
            writer.writerow([y for y in x])
    episode_data=[]

def plotter_avg(agent1reward,agent2reward,agent3reward):
    global agent1avg,agent2avg,agent3avg,turn
    agent1avg=(agent1avg+agent1reward)/turn
    agent2avg=(agent2avg+agent2reward)/turn
    agent3avg=(agent3avg+agent3reward)/turn
    turn=turn+1

def plot_episode():
    global episode,agent1avg,agent2avg,agent3avg,turn
    plot.plot(episode,agent1avg,'ro')
    plot.plot(episode,agent2avg,'go')
    plot.plot(episode,agent3avg,'bo')
    episode=episode+1
    agent1avg=0
    agent2avg=0
    agent3avg=0
    turn=0

#main program
checkers=checkersEnvironment()
agentOne=agentOne()
agentTwo=agentTwo()
agentThree=agentThree()

checkers.envInit()
fullState=checkers.envStart()

terminal=False
while True:
    while terminal==False:
        ###AGENT THREE ROUND###
        #set new states
        absoluteState3=agentThree.stateConcatThree(fullState,0)
        absoluteState2=agentTwo.stateConcatTwo(fullState)
        absoluteState1=agentOne.stateConcatOne(fullState)
        #get actions
        boardAction=agentThree.boardAction()
        messageAction=agentThree.messageAction()
        #!for record keeping!#
        agent3message=messageAction
        agent3action=boardAction
        #!end record keeping!#
        
        #get rewards
        (currentState, isTerminal, boardReward, generalReward)=checkers.envStepBoard(boardAction)
        (totalSpeakingReward, totalListeningReward)=checkers.envStepMessage(agentThree.msgStateHistory,agentThree.message_history_self,agentThree.action_history_self,agentThree.external_message_history)
        reward=[boardReward,generalReward,totalSpeakingReward,totalListeningReward]
        (oldAbsoluteState1,oldAbsoluteState2,oldAbsoluteState3, fullState)=checkers.prepNewState(messageAction,isTerminal,reward,currentState,absoluteState1,absoluteState2,absoluteState3,3)
        absoluteState3=agentThree.stateConcatThree(fullState,0)
        absoluteState2=agentTwo.stateConcatTwo(fullState)
        absoluteState1=agentOne.stateConcatOne(fullState)
        currentStateValue1=agentOne.stateValue(absoluteState1)
        currentStateValue2=agentTwo.stateValue(absoluteState2)
        oldStateValue1=agentOne.stateValue(oldAbsoluteState1)
        oldStateValue2=agentTwo.stateValue(oldAbsoluteState2)
        (relations,total_reward_3, boardRewardRegularized3, generalRewardRegularized3, totalSpeakingRewardRegularized3, totalListeningRewardRegularized3)=agentThree.updateParameters(oldAbsoluteState3,absoluteState3,currentStateValue1,currentStateValue2,oldStateValue1,oldStateValue2,reward)
        print("agent Three round done")
        ###AGENT ONE ROUND###
        checkers.updateState(relations)
        absoluteState3=agentThree.stateConcatThree(fullState,0)
        absoluteState2=agentTwo.stateConcatTwo(fullState)
        absoluteState1=agentOne.stateConcatOne(fullState)
        boardAction=agentOne.boardAction()
        messageAction=agentOne.messageAction()
        #!for record keeping!#
        agent1message=messageAction
        agent1action=boardAction
        #!end record keeping!#
        (currentState, isTerminal, boardReward, generalReward)=checkers.envStepBoard(boardAction)
        (totalSpeakingReward, totalListeningReward)=checkers.envStepMessage(agentOne.msgStateHistory,agentOne.message_history_self,agentOne.action_history_self,agentOne.external_message_history)
        reward=[boardReward,generalReward,totalSpeakingReward,totalListeningReward]
        (oldAbsoluteState1,oldAbsoluteState2,oldAbsoluteState3, fullState)=checkers.prepNewState(messageAction,isTerminal,reward,currentState,absoluteState1,absoluteState2,absoluteState3,1)
        (total_reward_1, boardRewardRegularized1, generalRewardRegularized1, totalSpeakingRewardRegularized1, totalListeningRewardRegularized1)=agentOne.updateParameters(oldAbsoluteState1,absoluteState1,reward)
        print("agent One round done")
        ###AGENT TWO ROUND###
        absoluteState3=agentThree.stateConcatThree(fullState,1)
        absoluteState2=agentTwo.stateConcatTwo(fullState)
        absoluteState1=agentOne.stateConcatOne(fullState)
        boardAction=agentTwo.boardAction()
        messageAction=agentTwo.messageAction()
        #!for record keeping!#
        agent2message=messageAction
        agent2action=boardAction
        #!end record keeping!#
        (currentState, isTerminal, boardReward, generalReward)=checkers.envStepBoard(boardAction)
        (totalSpeakingReward, totalListeningReward)=checkers.envStepMessage(agentTwo.msgStateHistory,agentTwo.message_history_self,agentTwo.action_history_self,agentTwo.external_message_history)
        reward=[boardReward,generalReward,totalSpeakingReward,totalListeningReward]
        (oldAbsoluteState1,oldAbsoluteState2,oldAbsoluteState3, fullState)=checkers.prepNewState(messageAction,isTerminal,reward,currentState,absoluteState1,absoluteState2,absoluteState3,2)
        (total_reward_2, boardRewardRegularized2, generalRewardRegularized2, totalSpeakingRewardRegularized2, totalListeningRewardRegularized2)=agentTwo.updateParameters(oldAbsoluteState2,absoluteState2,reward)
        print("agent Two round done")
        ### still needs to be finished
        turn_output(agent1message,agent2message,agent3message,agent1action,agent2action,agent3action,total_reward_1,total_reward_2,total_reward_3,[boardRewardRegularized1, generalRewardRegularized1, totalSpeakingRewardRegularized1, totalListeningRewardRegularized1],[boardRewardRegularized2, generalRewardRegularized2, totalSpeakingRewardRegularized2, totalListeningRewardRegularized2],[boardRewardRegularized3, generalRewardRegularized3, totalSpeakingRewardRegularized3, totalListeningRewardRegularized3])
        plotter_avg(total_reward_1,total_reward_2,total_reward_3)
        if isTerminal==True:
            terminal=True
    episode_csv()
    plot_episode()
    fullState=checkers.envStart()

    
