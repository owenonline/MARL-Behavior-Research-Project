import numpy as np
import matplotlib.pyplot as plot
import os
from rl_glue import RLGlue
from tqdm import tqdm

class checkersEnvironment():
    def envInit(self):
        self.terminal=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        
    def envStart(self):
        #reward is a 3x3 matrix. The first row is the message reward (based on how much the messages relate to the actions taken), the second row is the board reward (based on the legal movement of pieces; >=0), the third is general reward (based on winning or losing the game)
        reward=[[0,0,0],[0,0,0],[0,0,0]]
        #messages is a vector for 4 messages of length 0-100. message 1 is from agent 3 to agent 1, message 2 is from agent 3 to agent 2, message 3 is from agent 1 to agent 3, and message 4 is from agent 2 to agent 3
        messages=[list(0 for x in range(100)),list(0 for x in range(100)),list(0 for x in range(100)),list(0 for x in range(100))]
        relationVals=[0,0]
        #(y,x)
        boardState=[[[2,1],[1,2],[2,3],[1,4],[2,5],[1,6]],[[6,1],[5,2],[6,3],[5,4],[6,5],[5,6]]]
        isTerminal=False
        self.fullState=(reward,boardState,isTerminal,messages,relationVals)
        return self.fullState[1]
            
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

    def envStepMessage(self, messages, action, agentReal, piece):
        #use RIAL to determine message reward


class agentThree():
    def __init__(self):
        lstm_msg1_params=[]
        lstm_msg2_params=[]
        lstm_msg3_params=[]
        lstm_msg4_params=[]
        msg_ffn_params=[]
        value_fn_params=[]
        board_policy_params=[]
        msg_policy_params=[]
        
    def stateConcatThree(fullState):
        

class agentTwo():
    def __init__(self):
        lstm_msg1_params=[]
        lstm_msg2_params=[]
        lstm_msg3_params=[]
        lstm_msg4_params=[]
        msg_ffn_params=[]
        value_fn_params=[]
        board_policy_params=[]
        msg_policy_params=[]
        
    def stateConcatTwo(fullState):
        

class agentOne():
    def __init__(self):
        lstm_msg1_params=[]
        lstm_msg2_params=[]
        lstm_msg3_params=[]
        lstm_msg4_params=[]
        msg_ffn_params=[]
        value_fn_params=[]
        board_policy_params=[]
        msg_policy_params=[]
        
    def stateConcatOne(fullState):

def main():
    checkers=checkersEnvironment()
    agentOne=agentOne()
    agentTwo=agentTwo()
    agentThree=agentThree()
    
    checkers.envInit()
    checkers.envStart()
    


