import numpy as np
import matplotlib.pyplot as plot
import os
import numpy as np
import math
from random import randint
import csv
import tensorflow as tf

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

def tensor_conversion(x,lstm):
    arr=np.array(x)
    arr=arr.flatten()
    arr=np.float32(arr)
    if lstm==True:
        arr=np.expand_dims(np.expand_dims(arr, axis=0), axis=0)
    else:
        arr=np.expand_dims(arr, axis=0)
    arr=tf.convert_to_tensor(arr)
    return arr

def numpy_conversion(x):
    arr=x.numpy()
    arr=np.transpose(arr)
    arr=arr.flatten()
    return arr
    
class checkersEnvironment():
    def envInit(self):
        self.terminal=[[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]
        
    def envStart(self):
        #reward is a 3x3 matrix. The first row is the message reward (based on how much the messages relate to the actions taken), the second row is the board reward (based on the legal movement of pieces; >=0), the third is general reward (based on winning or losing the game)
        reward=[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
        #messages is a vector for 4 messages of length 0-100. message 1 is from agent 3 to agent 1, message 2 is from agent 3 to agent 2, message 3 is from agent 1 to agent 3, and message 4 is from agent 2 to agent 3
        messages=[list(0.0 for x in range(100)),list(0.0 for x in range(100)),list(0.0 for x in range(100)),list(0.0 for x in range(100))]
        #relation for agent 1 then 2
        relationVals=[0.0,0.0]
        #(y,x)
        boardState=[[[2.0,1.0],[1.0,2.0],[2.0,3.0],[1.0,4.0],[2.0,5.0],[1.0,6.0]],[[6.0,1],[5.0,2.0],[6.0,3.0],[5.0,4.0],[6.0,5.0],[5.0,6.0]]]
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
        return self.fullState


class agentThree():
    def __init__(self):
        ####################################TENSORFLOW VARIABLES####################################
        #lstms
        msg1=tf.keras.Input(shape=(1,100))
        lstm_msg1=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(msg1)
        self.lstm_msg1_model=tf.keras.Model(inputs=msg1,outputs=lstm_msg1)

        msg2=tf.keras.Input(shape=(1,100))
        lstm_msg2=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(msg2)
        self.lstm_msg2_model=tf.keras.Model(inputs=msg2,outputs=lstm_msg2)

        msg3=tf.keras.Input(shape=(1,100))
        lstm_msg3=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(msg3)
        self.lstm_msg3_model=tf.keras.Model(inputs=msg3,outputs=lstm_msg3)
        
        msg4=tf.keras.Input(shape=(1,100))
        lstm_msg4=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(msg4)
        self.lstm_msg4_model=tf.keras.Model(inputs=msg4,outputs=lstm_msg4)

        board=tf.keras.Input(shape=(1,24))
        lstm_board=tf.keras.layers.LSTM(200,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(board)
        self.lstm_board_model=tf.keras.Model(inputs=board,outputs=lstm_board)

        relation=tf.keras.Input(shape=(1,2))
        lstm_relation=tf.keras.layers.LSTM(50,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(relation)
        self.lstm_relation_model=tf.keras.Model(inputs=relation,outputs=lstm_relation)
        #ffns
        full_state=tf.keras.Input(shape=(1,1450))
        ffn_1=tf.keras.layers.Dense(1200,activation='relu',use_bias=True)(full_state)
        ffn_2=tf.keras.layers.Dense(950,activation='relu',use_bias=True)(ffn_1)
        ffn_3=tf.keras.layers.Dense(700,activation='relu',use_bias=True)(ffn_2)
        ffn_4=tf.keras.layers.Dense(450,activation='relu',use_bias=True)(ffn_3)
        ffn_5=tf.keras.layers.Dense(200,activation='relu',use_bias=True)(ffn_4)
        self.ffn_model=tf.keras.Model(inputs=full_state,outputs=ffn_5)
        #board policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        board_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true')(absolute_state)
        board_hidden_2=tf.keras.layers.Dense(200,activation='relu',use_bias='true')(board_hidden_1)
        board_hidden_3=tf.keras.layers.Dense(150,activation='relu',use_bias='true')(board_hidden_2)
        board_hidden_4=tf.keras.layers.Dense(100,activation='relu',use_bias='true')(board_hidden_3)
        board_hidden_5=tf.keras.layers.Dense(20,activation='relu',use_bias='true')(board_hidden_4)
        board_mu=tf.keras.layers.Dense(8,activation='relu',use_bias='true')(board_hidden_5)
        board_sigma=tf.keras.layers.Dense(8,activation='relu',use_bias='true')(board_hidden_5)
        self.board_model=tf.keras.Model(inputs=absolute_state,outputs=[board_mu,board_sigma])
        #message policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        message_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true')(absolute_state)
        message_hidden_2=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(message_hidden_1)
        message_hidden_3=tf.keras.layers.Dense(350,activation='relu',use_bias='true')(message_hidden_2)
        message_hidden_4=tf.keras.layers.Dense(400,activation='relu',use_bias='true')(message_hidden_3)
        message_hidden_5=tf.keras.layers.Dense(350,activation='relu',use_bias='true')(message_hidden_4)
        message_hidden_6=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(message_hidden_5)
        message_hidden_7=tf.keras.layers.Dense(250,activation='relu',use_bias='true')(message_hidden_6)
        message_mu=tf.keras.layers.Dense(200,activation='relu',use_bias='true')(message_hidden_7)
        message_sigma=tf.keras.layers.Dense(200,activation='relu',use_bias='true')(message_hidden_7)
        self.message_model=tf.keras.Model(inputs=absolute_state,outputs=[message_mu,message_sigma])
        #value function ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        value_hidden_1=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(absolute_state)
        value_hidden_2=tf.keras.layers.Dense(400,activation='relu',use_bias='true')(value_hidden_1)
        value_hidden_3=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(value_hidden_2)
        value_hidden_4=tf.keras.layers.Dense(200,activation='relu',use_bias='true')(value_hidden_3)
        value_hidden_5=tf.keras.layers.Dense(100,activation='relu',use_bias='true')(value_hidden_4)
        value_hidden_6=tf.keras.layers.Dense(1,activation='relu',use_bias='true')(value_hidden_5)
        self.value_model=tf.keras.Model(inputs=absolute_state,outputs=value_hidden_6)
        #value retention lists
        self.absolute_state_history=[]
        self.message_history_self=[np.zeros(200)]
        self.action_history_self=[np.zeros(8)]
        self.msgStateHistory=[]
        self.external_message_history=[]
        self.boardReward_history=[]
        self.generalReward_history=[]
        self.totalSpeakingReward_history=[]
        self.totalListeningReward_history=[]
        
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

        msg1_output=self.lstm_msg1_model(tensor_conversion(messageOne,True))
        msg2_output=self.lstm_msg2_model(tensor_conversion(messageTwo,True))
        msg3_output=self.lstm_msg3_model(tensor_conversion(messageThree,True))
        msg4_output=self.lstm_msg4_model(tensor_conversion(messageFour,True))
        board_output=self.lstm_board_model(tensor_conversion(boardState,True))
        relation_output=self.lstm_relation_model(tensor_conversion(relationVals,True))

        concat_state=np.concatenate((numpy_conversion(msg1_output),numpy_conversion(msg2_output),numpy_conversion(msg3_output),numpy_conversion(msg4_output),numpy_conversion(board_output),numpy_conversion(relation_output)))

        state=self.ffn_model(tensor_conversion(concat_state,False))

        state=numpy_conversion(state)

        self.absolute_state_history.append(state)
        
        return state

    def boardAction(self, state):
        (mean,std)=self.board_model(tensor_conversion(state,False))

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        action=tf.squeeze(norm_dist.sample(1),axis=0)

        action=numpy_conversion(action)
        
        if action[0]>-1.5:
            action[0]=-1
        else:
            action[0]=-2
        action[1]=np.round((5/(1+np.exp(-action[1]-5))))
        for x in range(len(action[2:])):
            action[x+2]=np.round((4/(1+np.exp(-(action[x+2]-4))))-2)

        boardAction=[action[0],action[1],action[2:]]
        self.action_history_self.append(action)
        return boardAction, norm_dist

    def messageAction(self, state):
        (mean,std)=self.message_model(tensor_conversion(state,False))

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        messages=tf.squeeze(norm_dist.sample(1),axis=0)

        messages=numpy_conversion(messages)
        
        message1=messages[0:100]
        message2=messages[100:200]
        messageAction=[message1,message2]
        
        self.message_history_self.append(messages)
        self.msgStateHistory.append(state)
        return messageAction, norm_dist

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

        return relations
        
class agentTwo():
    def __init__(self):
        ####################################TENSORFLOW VARIABLES####################################
        #lstms
        msg2=tf.keras.Input(shape=(1,100))
        lstm_msg2=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(msg2)
        self.lstm_msg2_model=tf.keras.Model(inputs=msg2,outputs=lstm_msg2)

        msg4=tf.keras.Input(shape=(1,100))
        lstm_msg4=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(msg4)
        self.lstm_msg4_model=tf.keras.Model(inputs=msg4,outputs=lstm_msg4)

        board=tf.keras.Input(shape=(1,24))
        lstm_board=tf.keras.layers.LSTM(200,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(board)
        self.lstm_board_model=tf.keras.Model(inputs=board,outputs=lstm_board)

        relation=tf.keras.Input(shape=(1,1))
        lstm_relation=tf.keras.layers.LSTM(50,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(relation)
        self.lstm_relation_model=tf.keras.Model(inputs=relation,outputs=lstm_relation)
        #ffns
        full_state=tf.keras.Input(shape=(1,850))
        ffn_1=tf.keras.layers.Dense(720,activation='relu',use_bias=True)(full_state)
        ffn_2=tf.keras.layers.Dense(590,activation='relu',use_bias=True)(ffn_1)
        ffn_3=tf.keras.layers.Dense(460,activation='relu',use_bias=True)(ffn_2)
        ffn_4=tf.keras.layers.Dense(330,activation='relu',use_bias=True)(ffn_3)
        ffn_5=tf.keras.layers.Dense(200,activation='relu',use_bias=True)(ffn_4)
        self.ffn_model=tf.keras.Model(inputs=full_state,outputs=ffn_5)
        #board policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        board_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true')(absolute_state)
        board_hidden_2=tf.keras.layers.Dense(200,activation='relu',use_bias='true')(board_hidden_1)
        board_hidden_3=tf.keras.layers.Dense(150,activation='relu',use_bias='true')(board_hidden_2)
        board_hidden_4=tf.keras.layers.Dense(100,activation='relu',use_bias='true')(board_hidden_3)
        board_hidden_5=tf.keras.layers.Dense(20,activation='relu',use_bias='true')(board_hidden_4)
        board_mu=tf.keras.layers.Dense(7,activation='relu',use_bias='true')(board_hidden_5)
        board_sigma=tf.keras.layers.Dense(7,activation='relu',use_bias='true')(board_hidden_5)
        self.board_model=tf.keras.Model(inputs=absolute_state,outputs=[board_mu,board_sigma])
        #message policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        message_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true')(absolute_state)
        message_hidden_2=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(message_hidden_1)
        message_hidden_3=tf.keras.layers.Dense(350,activation='relu',use_bias='true')(message_hidden_2)
        message_hidden_4=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(message_hidden_3)
        message_hidden_5=tf.keras.layers.Dense(250,activation='relu',use_bias='true')(message_hidden_4)
        message_hidden_6=tf.keras.layers.Dense(200,activation='relu',use_bias='true')(message_hidden_5)
        message_hidden_7=tf.keras.layers.Dense(150,activation='relu',use_bias='true')(message_hidden_6)
        message_mu=tf.keras.layers.Dense(100,activation='relu',use_bias='true')(message_hidden_7)
        message_sigma=tf.keras.layers.Dense(100,activation='relu',use_bias='true')(message_hidden_7)
        self.message_model=tf.keras.Model(inputs=absolute_state,outputs=[message_mu,message_sigma])
        #value function ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        value_hidden_1=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(absolute_state)
        value_hidden_2=tf.keras.layers.Dense(400,activation='relu',use_bias='true')(value_hidden_1)
        value_hidden_3=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(value_hidden_2)
        value_hidden_4=tf.keras.layers.Dense(200,activation='relu',use_bias='true')(value_hidden_3)
        value_hidden_5=tf.keras.layers.Dense(100,activation='relu',use_bias='true')(value_hidden_4)
        value_hidden_6=tf.keras.layers.Dense(1,activation='relu',use_bias='true')(value_hidden_5)
        self.value_model=tf.keras.Model(inputs=absolute_state,outputs=value_hidden_6)
        #value retention lists
        self.absolute_state_history=[]
        self.message_history_self=[np.zeros(100)]
        self.action_history_self=[np.zeros(8)]
        self.msgStateHistory=[]
        self.external_message_history=[]
        self.boardReward_history=[]
        self.generalReward_history=[]
        self.totalSpeakingReward_history=[]
        self.totalListeningReward_history=[]
        
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
        
        msg2_output=self.lstm_msg2_model(tensor_conversion(messageTwo,True))
        msg4_output=self.lstm_msg4_model(tensor_conversion(messageFour,True))
        board_output=self.lstm_board_model(tensor_conversion(boardState,True))
        relation_output=self.lstm_relation_model(tensor_conversion(relationVal,True))

        concat_state=np.concatenate((numpy_conversion(msg2_output),numpy_conversion(msg4_output),numpy_conversion(board_output),numpy_conversion(relation_output)))

        state=self.ffn_model(tensor_conversion(concat_state,False))

        state=numpy_conversion(state)

        self.absolute_state_history.append(state)
        return state

    def boardAction(self, state):
        (mean,std)=self.board_model(tensor_conversion(state,False))

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        action=tf.squeeze(norm_dist.sample(1),axis=0)

        action=numpy_conversion(action)

        action=np.insert(action,0,2)

        action[1]=np.round((5/(1+np.exp(-action[1]-5))))
        for x in range(len(action[2:])):
            action[x+2]=np.round((4/(1+np.exp(-(action[x+2]-4))))-2)
        boardAction=[action[0],action[1],action[2:]]
        self.action_history_self.append(action)
        return boardAction, norm_dist
        
    def messageAction(self, state):
        (mean,std)=self.message_model(tensor_conversion(state,False))

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        message4=tf.squeeze(norm_dist.sample(1),axis=0)

        message4=numpy_conversion(message4)
        
        messageAction=message4
        self.message_history_self.append(message4)
        self.msgStateHistory.append(state)
        return messageAction, norm_dist
    
    def stateValue(self, state):
        value=self.value_model(tensor_conversion(state,False))
        value=numpy_conversion(value)
        return value

    def updateParameters(self,oldAbsoluteState2,absoluteState2,reward):
        ####This code regularizes the reward:####
        self.boardReward_history.append(boardReward)
        self.generalReward_history.append(generalReward)
        self.totalSpeakingReward_history.append(totalSpeakingReward)
        self.totalListeningReward_history.append(totalListeningReward)

        
            
class agentOne():
    def __init__(self):
        ####################################TENSORFLOW VARIABLES####################################
        #lstms
        msg1=tf.keras.Input(shape=(1,100))
        lstm_msg1=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(msg1)
        self.lstm_msg1_model=tf.keras.Model(inputs=msg1,outputs=lstm_msg1)

        msg3=tf.keras.Input(shape=(1,100))
        lstm_msg3=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(msg3)
        self.lstm_msg3_model=tf.keras.Model(inputs=msg3,outputs=lstm_msg3)

        board=tf.keras.Input(shape=(1,24))
        lstm_board=tf.keras.layers.LSTM(200,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(board)
        self.lstm_board_model=tf.keras.Model(inputs=board,outputs=lstm_board)

        relation=tf.keras.Input(shape=(1,1))
        lstm_relation=tf.keras.layers.LSTM(50,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False)(relation)
        self.lstm_relation_model=tf.keras.Model(inputs=relation,outputs=lstm_relation)
        #ffns
        full_state=tf.keras.Input(shape=(1,850))
        ffn_1=tf.keras.layers.Dense(720,activation='relu',use_bias=True)(full_state)
        ffn_2=tf.keras.layers.Dense(590,activation='relu',use_bias=True)(ffn_1)
        ffn_3=tf.keras.layers.Dense(460,activation='relu',use_bias=True)(ffn_2)
        ffn_4=tf.keras.layers.Dense(330,activation='relu',use_bias=True)(ffn_3)
        ffn_5=tf.keras.layers.Dense(200,activation='relu',use_bias=True)(ffn_4)
        self.ffn_model=tf.keras.Model(inputs=full_state,outputs=ffn_5)
        #board policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        board_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true')(absolute_state)
        board_hidden_2=tf.keras.layers.Dense(200,activation='relu',use_bias='true')(board_hidden_1)
        board_hidden_3=tf.keras.layers.Dense(150,activation='relu',use_bias='true')(board_hidden_2)
        board_hidden_4=tf.keras.layers.Dense(100,activation='relu',use_bias='true')(board_hidden_3)
        board_hidden_5=tf.keras.layers.Dense(20,activation='relu',use_bias='true')(board_hidden_4)
        board_mu=tf.keras.layers.Dense(7,activation='relu',use_bias='true')(board_hidden_5)
        board_sigma=tf.keras.layers.Dense(7,activation='relu',use_bias='true')(board_hidden_5)
        self.board_model=tf.keras.Model(inputs=absolute_state,outputs=[board_mu,board_sigma])
        #message policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        message_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true')(absolute_state)
        message_hidden_2=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(message_hidden_1)
        message_hidden_3=tf.keras.layers.Dense(350,activation='relu',use_bias='true')(message_hidden_2)
        message_hidden_4=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(message_hidden_3)
        message_hidden_5=tf.keras.layers.Dense(250,activation='relu',use_bias='true')(message_hidden_4)
        message_hidden_6=tf.keras.layers.Dense(200,activation='relu',use_bias='true')(message_hidden_5)
        message_hidden_7=tf.keras.layers.Dense(150,activation='relu',use_bias='true')(message_hidden_6)
        message_mu=tf.keras.layers.Dense(100,activation='relu',use_bias='true')(message_hidden_7)
        message_sigma=tf.keras.layers.Dense(100,activation='relu',use_bias='true')(message_hidden_7)
        self.message_model=tf.keras.Model(inputs=absolute_state,outputs=[message_mu,message_sigma])
        #value function ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        value_hidden_1=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(absolute_state)
        value_hidden_2=tf.keras.layers.Dense(400,activation='relu',use_bias='true')(value_hidden_1)
        value_hidden_3=tf.keras.layers.Dense(300,activation='relu',use_bias='true')(value_hidden_2)
        value_hidden_4=tf.keras.layers.Dense(200,activation='relu',use_bias='true')(value_hidden_3)
        value_hidden_5=tf.keras.layers.Dense(100,activation='relu',use_bias='true')(value_hidden_4)
        value_hidden_6=tf.keras.layers.Dense(1,activation='relu',use_bias='true')(value_hidden_5)
        self.value_model=tf.keras.Model(inputs=absolute_state,outputs=value_hidden_6)
        #value retention lists
        self.absolute_state_history=[]
        self.message_history_self=[np.zeros(100)]
        self.action_history_self=[np.zeros(8)]
        self.msgStateHistory=[]
        self.external_message_history=[]
        self.boardReward_history=[]
        self.generalReward_history=[]
        self.totalSpeakingReward_history=[]
        self.totalListeningReward_history=[]
        #hyperparameters
        self.board_lr=0.00002
        self.message_lr=0.00002
        self.lstm_lr=0.00002
        self.ffn_lr=0.00002
        self.critic_lr=0.001
            
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

        msg1_output=self.lstm_msg1_model(tensor_conversion(messageOne,True))
        msg3_output=self.lstm_msg3_model(tensor_conversion(messageThree,True))
        board_output=self.lstm_board_model(tensor_conversion(boardState,True))
        relation_output=self.lstm_relation_model(tensor_conversion(relationVal,True))

        concat_state=np.concatenate((numpy_conversion(msg1_output),numpy_conversion(msg3_output),numpy_conversion(board_output),numpy_conversion(relation_output)))

        state=self.ffn_model(tensor_conversion(concat_state,False))

        state=numpy_conversion(state)

        self.absolute_state_history.append(state)
        return state

    def boardAction(self, state):
        (mean,std)=self.board_model(tensor_conversion(state,False))

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        action=tf.squeeze(norm_dist.sample(1),axis=0)

        action=numpy_conversion(action)

        action=np.insert(action,0,1)

        action[1]=np.round((5/(1+np.exp(-action[1]-5))))
        for x in range(len(action[2:])):
            action[x+2]=np.round((4/(1+np.exp(-(action[x+2]-4))))-2)
        boardAction=[action[0],action[1],action[2:]]
        self.action_history_self.append(action)
        return boardAction, norm_dist
        
    def messageAction(self, state):
        (mean,std)=self.message_model(tensor_conversion(state,False))

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        message3=tf.squeeze(norm_dist.sample(1),axis=0)

        message3=numpy_conversion(message3)
        
        messageAction=message3
        self.message_history_self.append(message3)
        self.msgStateHistory.append(state)
        return messageAction, norm_dist

    def stateValue(self, state):
        value=self.value_model(tensor_conversion(state,False))
        value=numpy_conversion(value)
        return value

    def updateParameters(self,oldAbsoluteState1,absoluteState1,reward):
        ####This code regularizes the reward:####
        self.boardReward_history.append(boardReward)
        self.generalReward_history.append(generalReward)
        self.totalSpeakingReward_history.append(totalSpeakingReward)
        self.totalListeningReward_history.append(totalListeningReward)

#main program initialization
checkers=checkersEnvironment()
agentOne=agentOne()
agentTwo=agentTwo()
agentThree=agentThree()

checkers.envInit()

#make startup variables here
fullState=checkers.envStart()
discount=0.99
terminal=False

episode_history=[]
while True:
    while terminal==False:
        ###AGENT THREE ROUND###
        #set new states
        absoluteState3=agentThree.stateConcatThree(fullState,0)
        absoluteState2=agentTwo.stateConcatTwo(fullState)
        absoluteState1=agentOne.stateConcatOne(fullState)
        #get actions
        (boardAction,normDistBoard)=agentThree.boardAction(absoluteState3)
        (messageAction,normDistMessage)=agentThree.messageAction(absoluteState3)
        #get rewards
        (currentState, isTerminal, boardReward, generalReward)=checkers.envStepBoard(boardAction)
        (totalSpeakingReward, totalListeningReward)=checkers.envStepMessage(agentThree.msgStateHistory,agentThree.message_history_self,agentThree.action_history_self,agentThree.external_message_history)
        reward=[boardReward,generalReward,totalSpeakingReward,totalListeningReward]
        #new state go get relationship vals
        (oldAbsoluteState1,oldAbsoluteState2,oldAbsoluteState3, fullState)=checkers.prepNewState(messageAction,isTerminal,reward,currentState,absoluteState1,absoluteState2,absoluteState3,3)
        absoluteState3=agentThree.stateConcatThree(fullState,0)
        absoluteState2=agentTwo.stateConcatTwo(fullState)
        absoluteState1=agentOne.stateConcatOne(fullState)
        currentStateValue1=agentOne.stateValue(absoluteState1)
        currentStateValue2=agentTwo.stateValue(absoluteState2)
        oldStateValue1=agentOne.stateValue(oldAbsoluteState1)
        oldStateValue2=agentTwo.stateValue(oldAbsoluteState2)
        #update parameters and output relations
        relations=agentThree.updateParameters(oldAbsoluteState3,absoluteState3,currentStateValue1,currentStateValue2,oldStateValue1,oldStateValue2,reward)
        print("agent 3 round done")
        
        ###AGENT ONE ROUND###
        #update states to incorporate new relationship values
        fullState=checkers.updateState(relations)
        absoluteState3=agentThree.stateConcatThree(fullState,0)
        absoluteState2=agentTwo.stateConcatTwo(fullState)
        absoluteState1=agentOne.stateConcatOne(fullState)
        #get actions
        (boardAction,normDistBoard)=agentOne.boardAction(absoluteState1)
        (messageAction,normDistMessage)=agentTwo.messageAction(absoluteState1)
        #get rewards
        (currentState, isTerminal, boardReward, generalReward)=checkers.envStepBoard(boardAction)
        (totalSpeakingReward, totalListeningReward)=checkers.envStepMessage(agentOne.msgStateHistory,agentOne.message_history_self,agentOne.action_history_self,agentOne.external_message_history)
        reward=[boardReward,generalReward,totalSpeakingReward,totalListeningReward]
        #new fullstate
        (oldAbsoluteState1,oldAbsoluteState2,oldAbsoluteState3, fullState)=checkers.prepNewState(messageAction,isTerminal,reward,currentState,absoluteState1,absoluteState2,absoluteState3,1)
        #update parameters
        print("agent 1 round done")

        ###AGENT TWO ROUND###
        #update states to account for agent 1's action
        absoluteState3=agentThree.stateConcatThree(fullState,1)
        absoluteState2=agentTwo.stateConcatTwo(fullState)
        absoluteState1=agentOne.stateConcatOne(fullState)
        #get actions
        (boardAction,normDistBoard)=agentTwo.boardAction(absoluteState2)
        (messageAction,normDistMessage)=agentTwo.messageAction(absoluteState2)
        #get rewards
        (currentState, isTerminal, boardReward, generalReward)=checkers.envStepBoard(boardAction)
        (totalSpeakingReward, totalListeningReward)=checkers.envStepMessage(agentTwo.msgStateHistory,agentTwo.message_history_self,agentTwo.action_history_self,agentTwo.external_message_history)
        reward=[boardReward,generalReward,totalSpeakingReward,totalListeningReward]
        #new fullstate
        (oldAbsoluteState1,oldAbsoluteState2,oldAbsoluteState3, fullState)=checkers.prepNewState(messageAction,isTerminal,reward,currentState,absoluteState1,absoluteState2,absoluteState3,2)
        #update parameters
        print("agent 2 round done")
        if isTerminal==True:
            terminal=True
    episode_history.append("placeholder")
    fullstate=checkers.envStart()
