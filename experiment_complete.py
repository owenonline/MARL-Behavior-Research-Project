import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf
import csv
from matplotlib.animation import FuncAnimation
import psutil
import collections

agent3_reward=[]
agent2_reward=[]
agent1_reward=[]

agent3=[]
agent2=[]
agent1=[]

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
    
class checkers_environment():
    def env_init(self):
        self.terminal=[[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]
        
    def env_start(self):
        #reward is a 3x3 matrix. The first row is the message reward (based on how much the messages relate to the actions taken), the second row is the board reward (based on the legal movement of pieces; >=0), the third is general reward (based on winning or losing the game)
        reward=[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
        #messages is a vector for 4 messages of length 0-100. message 1 is from agent 3 to agent 1, message 2 is from agent 3 to agent 2, message 3 is from agent 1 to agent 3, and message 4 is from agent 2 to agent 3
        messages=[list(0.0 for x in range(100)),list(0.0 for x in range(100)),list(0.0 for x in range(100)),list(0.0 for x in range(100))]
        #relation for agent 1 then 2
        relation_vals=[0.0,0.0]
        #(y,x)
        board_state=[[[2.0,1.0],[1.0,2.0],[2.0,3.0],[1.0,4.0],[2.0,5.0],[1.0,6.0]],[[6.0,1],[5.0,2.0],[6.0,3.0],[5.0,4.0],[6.0,5.0],[5.0,6.0]]]
        is_terminal=False
        self.full_state=(reward,board_state,is_terminal,messages,relation_vals)
        return self.full_state
            
    def env_step_board(self,board_action):
        agent_real=board_action[0]
        piece=int(board_action[1])
        action=board_action[2:]
        agent=int(abs(agent_real)-1)
        #agent_real=-1, -2, 1, or 2 to represent either agent 1, agent 2, or either agent but selected by agent 3 (the negative values). This is normalized to a binary value for the purpose of determining move legality.
        #agent: either 0 or 1 to represent agents 1 or 2
        #piece: any integer 0-5 to allow choice of any piece
        #action: [(-2-2), (-2-2), (-2-2), (-2-2), (-2-2), (-2-2)] with those being the movement increments. Any single move in checkers is either 0, 1, or 2 spaces in length hence the (-2-2). a negative value moves left diagonal
        #this gets the state from the tuple
        last_state=self.full_state[1] 

        current_state=last_state#sets the current state to the most updated state before the loop
        board_reward=0
        general_reward=0
        is_terminal=False
        
        for x in range(len(action)):
            piece_start=current_state[int(agent)][int(piece)] #this gets the situation of the piece at the start of the loop  
            current_action=int(action[0][x])

            if piece_start==[0,0]:
                #this means an already eliminated piece has been selected
                board_reward+=-10
                break
            
            #checks if the move was legal
            if current_action==0:
                if x==0:
                    #not moving any piece on the first move of the turn is illegal
                    board_reward+=-10
                #a move of 0 always ends the turn
                break
                
            
            #checks that the new index is within the board and is unoccupied by either an opposing or friendly piece
            elif 1<=piece_start[0]+abs(current_action)<6 and 1<piece_start[1]+current_action<=6 and [piece_start[0]+abs(current_action),piece_start[1]+current_action] not in current_state[0] and [piece_start[0]+abs(current_action),piece_start[1]+current_action] not in current_state[1]:
                #if no move to capture was made and the previous check was passed, the move is legal and the current state can be updated to reflect that.
                if current_action==1 or current_action==-1:
                    current_state[agent][piece]=[piece_start[0]+abs(current_action),piece_start[1]+current_action]
                    break
                elif current_action==2 or current_action==-2:
                    #this just checks if the jumped over square contains an enemy piece. this must be true for the move to be legal
                    if [piece_start[0]+abs(current_action)-1,piece_start[1]+((abs(current_action)-1)*(current_action/abs(current_action)))] in last_state[-1*agent+1]:
                        current_state[agent][piece]=[piece_start[0]+abs(current_action),piece_start[1]+current_action]
                        #sets the state of the enemy piece that was taken to 0
                        current_state[-1*agent+1][last_state[-1*agent+1].index([piece_start[0]+abs(current_action)-1,piece_start[1]+((abs(current_action)-1)*(current_action/abs(current_action)))])]=[0,0]
                        #sets piece_start for the starting position of the next part of the move
                        piece_start=current_state[agent][piece]
                    else:
                        #the agent used a jump move without eliminating an enemy piece, which is an illegal move
                        board_reward+=-10
                        break
                else:
                    #invalid move length
                    board_reward+=-10
                    break
            else:
                #this means the move selected is illegal because it lands on another piece or goes off the board
                board_reward+=-10
                break
            

        if (current_state[0]==self.terminal and agent_real==1) or (current_state[1]==self.terminal and agent_real==2):
            #agents 1 and 2 get a -100 general reward for losing
            general_reward=-100
            is_terminal=True
        elif (current_state[1]==self.terminal and agent_real==1) or (current_state[0]==self.terminal and agent_real==2):
            #agents 1 and 2 get a 100 general reward for winning
            general_reward=100
            is_terminal=True
        elif (agent_real==-1 or agent_real==-2) and (current_state[0]==self.terminal or current_state[1]==self.terminal):
            #agent 3 gets a -100 general reward when the game ends
            general_reward=-100
            is_terminal=True
        elif (agent_real==-1 or agent_real==-2) and not (current_state[0]==self.terminal or current_state[1]==self.terminal):
            #agent 3 gets a +1 general reward every time step
            general_reward=1
        elif (agent_real==1 or agent_real==2) and not (current_state[0]==self.terminal or current_state[1]==self.terminal):
            #agents 1 and 2 get a -1 general reward every time step
            general_reward=-1

        return (current_state, is_terminal, board_reward, general_reward)

    def env_step_message(self,state_history,message_history,action_history,external_message_history):
        #lambda_var is a hyperparameter to weight the error
        lambda_var=10
        total_speaking_reward=0
        for x in range(len(state_history)-1):
            msg_diff=np.absolute(np.subtract(message_history[-1],message_history[x])).mean()
            state_diff=np.absolute(np.subtract(state_history[-1],state_history[x])).mean()

            msg_diff=1/(1-np.exp(-(msg_diff-4)))
            state_diff=1/(1-np.exp(-(state_diff-4)))

            temp=0
            if msg_diff>state_diff:
                temp=msg_diff/state_diff
            else:
                temp=state_diff/msg_diff

            total_speaking_reward=total_speaking_reward+(-np.absolute(temp)*lambda_var)
        total_speaking_reward=total_speaking_reward/len(state_history)
        #now it's time to get the listening reward.
        total_listening_reward=0
        for x in range(len(external_message_history)-1):
            action_diff=np.absolute(np.subtract(action_history[-1],action_history[x])).mean()
            msg_diff=np.absolute(np.subtract(external_message_history[-1],external_message_history[x])).mean()

            action_diff=1/(1-np.exp(-(action_diff-4)))
            msg_diff=1/(1-np.exp(-(msg_diff-4)))

            temp=0
            if msg_diff>action_diff:
                temp=msg_diff/action_diff
            else:
                temp=action_diff/msg_diff

            total_listening_reward=total_listening_reward+(-np.absolute(temp)*lambda_var)
        total_listening_reward=total_listening_reward/len(external_message_history)
        #to prevent it from being 0 at the start and give a lil motivation imma make it slightly negative
        total_listening_reward=total_listening_reward-0.001
        total_speaking_reward=total_speaking_reward-0.001
        return (total_speaking_reward, total_listening_reward)

    def prep_new_state(self,message_action,is_terminal_new,reward_new,current_state,absolute_state_1,absolute_state_2,absolute_state_3,agent):
        [reward,board_state,is_terminal,messages,relation_vals]=self.full_state
        board_state=current_state
        is_terminal=is_terminal_new
        [a,b,c,d]=messages
        if agent==3:
            reward[2]=reward_new
            [a,b]=message_action
        if agent==1:
            reward[0]=reward_new
            c=message_action
        if agent==2:
            reward[1]=reward_new
            d=message_action
        messages=[a,b,c,d]
        self.full_state=[reward,board_state,is_terminal,messages,relation_vals]
        old_absolute_state_1=absolute_state_1
        old_absolute_state_2=absolute_state_2
        old_absolute_state_3=absolute_state_3
        return (old_absolute_state_1,old_absolute_state_2,old_absolute_state_3, self.full_state)

    def updateState(self, relation_values):
        newVals=[]
        for y in relation_values:
            newVals.append(y[0])
        self.full_state[-1]=newVals
        return self.full_state


class agent_three():
    def __init__(self):
        ####################################TENSORFLOW VARIABLES####################################
        #lstms
        msg1=tf.keras.Input(shape=(1,100))
        lstm_msg1=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(msg1)
        self.lstm_msg1_model=tf.keras.Model(inputs=msg1,outputs=lstm_msg1)

        msg2=tf.keras.Input(shape=(1,100))
        lstm_msg2=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(msg2)
        self.lstm_msg2_model=tf.keras.Model(inputs=msg2,outputs=lstm_msg2)

        msg3=tf.keras.Input(shape=(1,100))
        lstm_msg3=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(msg3)
        self.lstm_msg3_model=tf.keras.Model(inputs=msg3,outputs=lstm_msg3)
        
        msg4=tf.keras.Input(shape=(1,100))
        lstm_msg4=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(msg4)
        self.lstm_msg4_model=tf.keras.Model(inputs=msg4,outputs=lstm_msg4)

        board=tf.keras.Input(shape=(1,24))
        lstm_board=tf.keras.layers.LSTM(200,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board)
        self.lstm_board_model=tf.keras.Model(inputs=board,outputs=lstm_board)

        relation=tf.keras.Input(shape=(1,2))
        lstm_relation=tf.keras.layers.LSTM(50,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(relation)
        self.lstm_relation_model=tf.keras.Model(inputs=relation,outputs=lstm_relation)
        #ffns
        full_state=tf.keras.Input(shape=(1,1450))
        ffn_1=tf.keras.layers.Dense(1200,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(full_state)
        ffn_2=tf.keras.layers.Dense(950,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_1)
        ffn_3=tf.keras.layers.Dense(700,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_2)
        ffn_4=tf.keras.layers.Dense(450,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_3)
        ffn_5=tf.keras.layers.Dense(200,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_4)
        self.ffn_model=tf.keras.Model(inputs=full_state,outputs=ffn_5)
        #board policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        board_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(absolute_state)
        board_hidden_2=tf.keras.layers.Dense(200,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_1)
        board_hidden_3=tf.keras.layers.Dense(150,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_2)
        board_hidden_4=tf.keras.layers.Dense(100,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_3)
        board_hidden_5=tf.keras.layers.Dense(20,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_4)
        board_mu=tf.keras.layers.Dense(8,activation=None,use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_5)
        board_sigma=tf.keras.layers.Dense(8,activation='exponential',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_5)
        self.board_model=tf.keras.Model(inputs=absolute_state,outputs=[board_mu,board_sigma])
        #message policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        message_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(absolute_state)
        message_hidden_2=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_1)
        message_hidden_3=tf.keras.layers.Dense(350,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_2)
        message_hidden_4=tf.keras.layers.Dense(400,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_3)
        message_hidden_5=tf.keras.layers.Dense(350,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_4)
        message_hidden_6=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_5)
        message_hidden_7=tf.keras.layers.Dense(250,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_6)
        message_mu=tf.keras.layers.Dense(200,activation=None,use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_7)
        message_sigma=tf.keras.layers.Dense(200,activation='exponential',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_7)
        self.message_model=tf.keras.Model(inputs=absolute_state,outputs=[message_mu,message_sigma])
        #value function ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        value_hidden_1=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(absolute_state)
        value_hidden_2=tf.keras.layers.Dense(400,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_1)
        value_hidden_3=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_2)
        value_hidden_4=tf.keras.layers.Dense(200,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_3)
        value_hidden_5=tf.keras.layers.Dense(100,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_4)
        value_hidden_6=tf.keras.layers.Dense(1,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_5)
        self.value_model=tf.keras.Model(inputs=absolute_state,outputs=value_hidden_6)
        #value retention lists
        self.absolute_state_history=[]
        self.message_history_self=[np.zeros(200)]
        self.action_history_self=[np.zeros(8)]
        self.orig_action_history=[]
        self.orig_message_history=[]
        self.msgstate_history=[]
        self.external_message_history=[]
        self.board_reward_history=[]
        self.general_reward_history=[]
        self.total_speaking_reward_history=[]
        self.total_listening_reward_history=[]
        #optimization objects
        self.lambda_val=0.001
        self.value_lr=0.00002
        self.board_lr=0.00002
        self.message_lr=0.00002
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        
    def state_concat_three(self, full_state, msg):
        reward=full_state[2]
        board_state=full_state[1]
        temp=[]
        for x in board_state:
            for y in x:
                for z in y:
                    temp.append(z)
        board_state=temp
        is_terminal=full_state[2]
        messageOne=full_state[3][0]
        messageTwo=full_state[3][1]
        messageThree=full_state[3][2]
        messageFour=full_state[3][3]
        relation_vals=full_state[4]
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
        board_output=self.lstm_board_model(tensor_conversion(board_state,True))
        relation_output=self.lstm_relation_model(tensor_conversion(relation_vals,True))

        concat_state=tf.concat([msg1_output,msg2_output,msg3_output,msg4_output,board_output,relation_output],1)

        state=self.ffn_model(tf.expand_dims(concat_state,0))

        self.absolute_state_history.append(state)
        
        return state

    def board_action(self, state):
        (mean,std)=self.board_model(state)

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        action=tf.squeeze(norm_dist.sample(1),axis=0)

        self.orig_action_history.append(action)

        action=numpy_conversion(action)
        action=np.nan_to_num(action)
        
        if action[0]>-1.5:
            action[0]=-1
        else:
            action[0]=-2

        if action[1]>=5:
            action[1]=5
        if 5>action[1]>=4:
            action[1]=4
        if 4>action[1]>=3:
            action[1]=3
        if 3>action[1]>=2:
            action[1]=2
        if 2>action[1]:
            action[1]=1
        
        for x in range(len(action[2:])):
            if action[x+2]>=2:
                action[x+2]=2
            if 2>action[x+2]>=1:
                action[x+2]=1
            if 1>action[x+2]>=0:
                action[x+2]=0
            if 0>action[x+2]>=-1:
                action[x+2]=-1
            if -2>action[x+2]:
                action[x+2]=-2

        board_action=[action[0],action[1],action[2:]]
        self.action_history_self.append(action)
        return board_action, norm_dist

    def message_action(self, state):
        (mean,std)=self.message_model(state)

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        messages=tf.squeeze(norm_dist.sample(1),axis=0)

        self.orig_message_history.append(messages)
        
        messages=numpy_conversion(messages)
        
        message1=messages[0:100]
        message2=messages[100:200]
        message_action=[message1,message2]
        
        self.message_history_self.append(messages)
        self.msgstate_history.append(state)
        return message_action, norm_dist

    def state_value(self, state):
        value=self.value_model(state)
        return value

    def updateParameters(self,old_absolute_state_3,absolute_state_3,current_state_value1,current_state_value2,old_state_value_1,old_state_value_2,reward,action,message,normdist_board,normdist_message):
        ####This code adds the relation values to the general reward####
        (board_reward,general_reward,total_speaking_reward,total_listening_reward)=reward
        relations=checkers.full_state[-1]
        print(relations)
        #relation discounting amount
        discountFactor=0.1
        relations[0]+=(current_state_value1-old_state_value_1)*discountFactor
        relations[1]+=(current_state_value2-old_state_value_2)*discountFactor
        if (relations[0]-checkers.full_state[-1][0]) < (relations[0]-checkers.full_state[-1][0]):
            general_reward+=relations[0]-checkers.full_state[-1][0]
        else:
            general_reward+=relations[1]-checkers.full_state[-1][1]

        ####This code regularizes the reward:####
        self.board_reward_history.append(board_reward)
        self.general_reward_history.append(general_reward)
        self.total_speaking_reward_history.append(total_speaking_reward)
        self.total_listening_reward_history.append(total_listening_reward)

        overall_reward=board_reward+general_reward+total_speaking_reward+total_listening_reward
        overall_board_reward=board_reward+general_reward
        overall_message_reward=general_reward+total_speaking_reward+total_listening_reward

        value_error_target=overall_reward+(self.lambda_val*self.state_value(absolute_state_3))
        value_error_overall=(overall_reward+(self.lambda_val*self.state_value(absolute_state_3)))-self.state_value(old_absolute_state_3)
        value_error_board=(overall_board_reward+(self.lambda_val*self.state_value(absolute_state_3)))-self.state_value(old_absolute_state_3)
        value_error_message=(overall_message_reward+(self.lambda_val*self.state_value(absolute_state_3)))-self.state_value(old_absolute_state_3)

        loss_board=-tf.math.reduce_sum(tf.math.log(normdist_board.prob(self.orig_action_history[-1]))*value_error_board)
        loss_message=-tf.math.reduce_sum(tf.math.log(normdist_message.prob(self.orig_message_history[-1]))*value_error_message)
        loss_critic=self.huber_loss(self.state_value(old_absolute_state_3),value_error_target)

        #there is an issue with critic grads and gradient pass through, but everything else seems to be working fine. I also think I don't need the second and third tape layers

        board_grads=tape.gradient(loss_board,self.board_model.trainable_variables)
        message_grads=tape.gradient(loss_message,self.message_model.trainable_variables)
        critic_grads=tape.gradient(loss_critic,self.value_model.trainable_variables)
        ffn_grads=tape.gradient(critic_grads,self.ffn_model.trainable_variables)

        msg1_grads=tape.gradient(ffn_grads,self.lstm_msg1_model.trainable_variables)
        msg2_grads=tape.gradient(ffn_grads,self.lstm_msg2_model.trainable_variables)
        msg3_grads=tape.gradient(ffn_grads,self.lstm_msg3_model.trainable_variables)
        msg4_grads=tape.gradient(ffn_grads,self.lstm_msg4_model.trainable_variables)
        board_lstm_grads=tape.gradient(ffn_grads,self.lstm_board_model.trainable_variables)
        relation_grads=tape.gradient(ffn_grads,self.lstm_relation_model.trainable_variables)

        self.optimizer.apply_gradients(zip(board_grads,self.board_model.trainable_variables))
        self.optimizer.apply_gradients(zip(message_grads,self.message_model.trainable_variables))
        self.optimizer.apply_gradients(zip(critic_grads,self.value_model.trainable_variables))
        self.optimizer.apply_gradients(zip(ffn_grads,self.ffn_model.trainable_variables))
        self.optimizer.apply_gradients(zip(msg1_grads,self.lstm_msg1_model.trainable_variables))
        self.optimizer.apply_gradients(zip(msg2_grads,self.lstm_msg2_model.trainable_variables))
        self.optimizer.apply_gradients(zip(msg3_grads,self.lstm_msg3_model.trainable_variables))
        self.optimizer.apply_gradients(zip(msg4_grads,self.lstm_msg4_model.trainable_variables))
        self.optimizer.apply_gradients(zip(board_lstm_grads,self.lstm_board_model.trainable_variables))
        self.optimizer.apply_gradients(zip(relation_grads,self.lstm_relation_model.trainable_variables))

        agent3_low.append([message,action,old_absolute_state_3,absolute_state_3,reward[0],reward[1],reward[2],reward[3],relations])

        return relations, numpy_conversion(overall_reward)
        
class agent_two():
    def __init__(self):
        ####################################TENSORFLOW VARIABLES####################################
        #lstms
        msg2=tf.keras.Input(shape=(1,100))
        lstm_msg2=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(msg2)
        self.lstm_msg2_model=tf.keras.Model(inputs=msg2,outputs=lstm_msg2)

        msg4=tf.keras.Input(shape=(1,100))
        lstm_msg4=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(msg4)
        self.lstm_msg4_model=tf.keras.Model(inputs=msg4,outputs=lstm_msg4)

        board=tf.keras.Input(shape=(1,24))
        lstm_board=tf.keras.layers.LSTM(200,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board)
        self.lstm_board_model=tf.keras.Model(inputs=board,outputs=lstm_board)

        relation=tf.keras.Input(shape=(1,1))
        lstm_relation=tf.keras.layers.LSTM(50,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(relation)
        self.lstm_relation_model=tf.keras.Model(inputs=relation,outputs=lstm_relation)
        #ffns
        full_state=tf.keras.Input(shape=(1,850))
        ffn_1=tf.keras.layers.Dense(720,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(full_state)
        ffn_2=tf.keras.layers.Dense(590,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_1)
        ffn_3=tf.keras.layers.Dense(460,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_2)
        ffn_4=tf.keras.layers.Dense(330,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_3)
        ffn_5=tf.keras.layers.Dense(200,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_4)
        self.ffn_model=tf.keras.Model(inputs=full_state,outputs=ffn_5)
        #board policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        board_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(absolute_state)
        board_hidden_2=tf.keras.layers.Dense(200,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_1)
        board_hidden_3=tf.keras.layers.Dense(150,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_2)
        board_hidden_4=tf.keras.layers.Dense(100,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_3)
        board_hidden_5=tf.keras.layers.Dense(20,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_4)
        board_mu=tf.keras.layers.Dense(7,activation=None,use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_5)
        board_sigma=tf.keras.layers.Dense(7,activation='exponential',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_5)
        self.board_model=tf.keras.Model(inputs=absolute_state,outputs=[board_mu,board_sigma])
        #message policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        message_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(absolute_state)
        message_hidden_2=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_1)
        message_hidden_3=tf.keras.layers.Dense(350,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_2)
        message_hidden_4=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_3)
        message_hidden_5=tf.keras.layers.Dense(250,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_4)
        message_hidden_6=tf.keras.layers.Dense(200,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_5)
        message_hidden_7=tf.keras.layers.Dense(150,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_6)
        message_mu=tf.keras.layers.Dense(100,activation=None,use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_7)
        message_sigma=tf.keras.layers.Dense(100,activation='exponential',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_7)
        self.message_model=tf.keras.Model(inputs=absolute_state,outputs=[message_mu,message_sigma])
        #value function ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        value_hidden_1=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(absolute_state)
        value_hidden_2=tf.keras.layers.Dense(400,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_1)
        value_hidden_3=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_2)
        value_hidden_4=tf.keras.layers.Dense(200,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_3)
        value_hidden_5=tf.keras.layers.Dense(100,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_4)
        value_hidden_6=tf.keras.layers.Dense(1,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_5)
        self.value_model=tf.keras.Model(inputs=absolute_state,outputs=value_hidden_6)
        #value retention lists
        self.absolute_state_history=[]
        self.message_history_self=[np.zeros(100)]
        self.action_history_self=[np.zeros(8)]
        self.orig_action_history=[]
        self.orig_message_history=[]
        self.msgstate_history=[]
        self.external_message_history=[]
        self.board_reward_history=[]
        self.general_reward_history=[]
        self.total_speaking_reward_history=[]
        self.total_listening_reward_history=[]
        #optimization objects
        self.lambda_val=0.001
        self.value_lr=0.00002
        self.board_lr=0.00002
        self.message_lr=0.00002
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        
    def state_concat_two(self, full_state):
        reward=full_state[0]
        board_state=full_state[1]
        temp=[]
        for x in board_state:
            for y in x:
                for z in y:
                    temp.append(z)
        board_state=temp
        is_terminal=full_state[2]
        messageTwo=full_state[3][1]
        messageFour=full_state[3][3]
        relationVal=[full_state[4][1]]
        x=0
        for y in self.external_message_history:
            if (y==messageTwo).all():
                x=1
        if x==0:
            self.external_message_history.append(messageTwo)
        
        msg2_output=self.lstm_msg2_model(tensor_conversion(messageTwo,True))
        msg4_output=self.lstm_msg4_model(tensor_conversion(messageFour,True))
        board_output=self.lstm_board_model(tensor_conversion(board_state,True))
        relation_output=self.lstm_relation_model(tensor_conversion(relationVal,True))

        concat_state=tf.concat([msg2_output,msg4_output,board_output,relation_output],1)

        state=self.ffn_model(tf.expand_dims(concat_state,0))

        self.absolute_state_history.append(state)
        return state

    def board_action(self, state):
        (mean,std)=self.board_model(state)

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        action=tf.squeeze(norm_dist.sample(1),axis=0)
        

        self.orig_action_history.append(action)

        action=numpy_conversion(action)
        action=np.nan_to_num(action)

        action=np.insert(action,0,2)

        if action[1]>=5:
            action[1]=5
        if 5>action[1]>=4:
            action[1]=4
        if 4>action[1]>=3:
            action[1]=3
        if 3>action[1]>=2:
            action[1]=2
        if 2>action[1]:
            action[1]=1
        
        for x in range(len(action[2:])):
            if action[x+2]>=2:
                action[x+2]=2
            if 2>action[x+2]>=1:
                action[x+2]=1
            if 1>action[x+2]>=0:
                action[x+2]=0
            if 0>action[x+2]>=-1:
                action[x+2]=-1
            if -2>action[x+2]:
                action[x+2]=-2
                
        board_action=[action[0],action[1],action[2:]]
        self.action_history_self.append(action)
        return board_action, norm_dist
        
    def message_action(self, state):
        (mean,std)=self.message_model(state)

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        message4=tf.squeeze(norm_dist.sample(1),axis=0)

        self.orig_message_history.append(message4)

        message4=numpy_conversion(message4)
        
        message_action=message4
        self.message_history_self.append(message4)
        self.msgstate_history.append(state)
        return message_action, norm_dist
    
    def state_value(self, state):
        value=self.value_model(state)
        return value

    def updateParameters(self,old_absolute_state_2,absolute_state_2,reward,action,message,normdist_board,normdist_message):
        ####This code regularizes the reward:####
        (board_reward,general_reward,total_speaking_reward,total_listening_reward)=reward
        self.board_reward_history.append(board_reward)
        self.general_reward_history.append(general_reward)
        self.total_speaking_reward_history.append(total_speaking_reward)
        self.total_listening_reward_history.append(total_listening_reward)

        overall_reward=board_reward+general_reward+total_speaking_reward+total_listening_reward
        overall_board_reward=board_reward+general_reward
        overall_message_reward=general_reward+total_speaking_reward+total_listening_reward

        value_error_target=overall_reward+(self.lambda_val*self.state_value(absolute_state_2))
        value_error_overall=(overall_reward+(self.lambda_val*self.state_value(absolute_state_2)))-self.state_value(old_absolute_state_2)
        value_error_board=(overall_board_reward+(self.lambda_val*self.state_value(absolute_state_2)))-self.state_value(old_absolute_state_2)
        value_error_message=(overall_message_reward+(self.lambda_val*self.state_value(absolute_state_2)))-self.state_value(old_absolute_state_2)

        loss_board=-tf.math.reduce_sum(tf.math.log(normdist_board.prob(self.orig_action_history[-1]))*value_error_board)
        loss_message=-tf.math.reduce_sum(tf.math.log(normdist_message.prob(self.orig_message_history[-1]))*value_error_message)
        loss_critic=self.huber_loss(self.state_value(old_absolute_state_2),value_error_target)

        board_grads=tape.gradient(loss_board,self.board_model.trainable_variables)
        message_grads=tape.gradient(loss_message,self.message_model.trainable_variables)
        critic_grads=tape.gradient(loss_critic,self.value_model.trainable_variables)
        ffn_grads=tape.gradient(critic_grads,self.ffn_model.trainable_variables)

        msg2_grads=tape.gradient(ffn_grads,self.lstm_msg2_model.trainable_variables)
        msg4_grads=tape.gradient(ffn_grads,self.lstm_msg4_model.trainable_variables)
        board_lstm_grads=tape.gradient(ffn_grads,self.lstm_board_model.trainable_variables)
        relation_grads=tape.gradient(ffn_grads,self.lstm_relation_model.trainable_variables)

        self.optimizer.apply_gradients(zip(board_grads,self.board_model.trainable_variables))
        self.optimizer.apply_gradients(zip(message_grads,self.message_model.trainable_variables))
        self.optimizer.apply_gradients(zip(critic_grads,self.value_model.trainable_variables))
        self.optimizer.apply_gradients(zip(ffn_grads,self.ffn_model.trainable_variables))
        self.optimizer.apply_gradients(zip(msg2_grads,self.lstm_msg2_model.trainable_variables))
        self.optimizer.apply_gradients(zip(msg4_grads,self.lstm_msg4_model.trainable_variables))
        self.optimizer.apply_gradients(zip(board_lstm_grads,self.lstm_board_model.trainable_variables))
        self.optimizer.apply_gradients(zip(relation_grads,self.lstm_relation_model.trainable_variables))

        agent2_low.append([message,action,old_absolute_state_2,absolute_state_2,reward[0],reward[1],reward[2],reward[3]])

        return overall_reward

            
class agent_one():
    def __init__(self):
        ####################################TENSORFLOW VARIABLES####################################
        #lstms
        msg1=tf.keras.Input(shape=(1,100))
        lstm_msg1=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(msg1)
        self.lstm_msg1_model=tf.keras.Model(inputs=msg1,outputs=lstm_msg1)

        msg3=tf.keras.Input(shape=(1,100))
        lstm_msg3=tf.keras.layers.LSTM(300,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(msg3)
        self.lstm_msg3_model=tf.keras.Model(inputs=msg3,outputs=lstm_msg3)

        board=tf.keras.Input(shape=(1,24))
        lstm_board=tf.keras.layers.LSTM(200,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board)
        self.lstm_board_model=tf.keras.Model(inputs=board,outputs=lstm_board)

        relation=tf.keras.Input(shape=(1,1))
        lstm_relation=tf.keras.layers.LSTM(50,activation='tanh',recurrent_activation='sigmoid',use_bias=True,unroll=False,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(relation)
        self.lstm_relation_model=tf.keras.Model(inputs=relation,outputs=lstm_relation)
        #ffns
        full_state=tf.keras.Input(shape=(1,850))
        ffn_1=tf.keras.layers.Dense(720,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(full_state)
        ffn_2=tf.keras.layers.Dense(590,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_1)
        ffn_3=tf.keras.layers.Dense(460,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_2)
        ffn_4=tf.keras.layers.Dense(330,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_3)
        ffn_5=tf.keras.layers.Dense(200,activation='relu',use_bias=True,kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(ffn_4)
        self.ffn_model=tf.keras.Model(inputs=full_state,outputs=ffn_5)
        #board policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        board_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(absolute_state)
        board_hidden_2=tf.keras.layers.Dense(200,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_1)
        board_hidden_3=tf.keras.layers.Dense(150,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_2)
        board_hidden_4=tf.keras.layers.Dense(100,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_3)
        board_hidden_5=tf.keras.layers.Dense(20,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_4)
        board_mu=tf.keras.layers.Dense(7,activation=None,use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_5)
        board_sigma=tf.keras.layers.Dense(7,activation='exponential',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(board_hidden_5)
        self.board_model=tf.keras.Model(inputs=absolute_state,outputs=[board_mu,board_sigma])
        #message policy ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        message_hidden_1=tf.keras.layers.Dense(250,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(absolute_state)
        message_hidden_2=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_1)
        message_hidden_3=tf.keras.layers.Dense(350,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_2)
        message_hidden_4=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_3)
        message_hidden_5=tf.keras.layers.Dense(250,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_4)
        message_hidden_6=tf.keras.layers.Dense(200,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_5)
        message_hidden_7=tf.keras.layers.Dense(150,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_6)
        message_mu=tf.keras.layers.Dense(100,activation=None,use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_7)
        message_sigma=tf.keras.layers.Dense(100,activation='exponential',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(message_hidden_7)
        self.message_model=tf.keras.Model(inputs=absolute_state,outputs=[message_mu,message_sigma])
        #value function ffns
        absolute_state=tf.keras.Input(shape=(1,200))
        value_hidden_1=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(absolute_state)
        value_hidden_2=tf.keras.layers.Dense(400,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_1)
        value_hidden_3=tf.keras.layers.Dense(300,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_2)
        value_hidden_4=tf.keras.layers.Dense(200,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_3)
        value_hidden_5=tf.keras.layers.Dense(100,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_4)
        value_hidden_6=tf.keras.layers.Dense(1,activation='relu',use_bias='true',kernel_regularizer='l2',bias_regularizer='l2',activity_regularizer='l2')(value_hidden_5)
        self.value_model=tf.keras.Model(inputs=absolute_state,outputs=value_hidden_6)
        #value retention lists
        self.absolute_state_history=[]
        self.message_history_self=[np.zeros(100)]
        self.action_history_self=[np.zeros(8)]
        self.orig_action_history=[]
        self.orig_message_history=[]
        self.msgstate_history=[]
        self.external_message_history=[]
        self.board_reward_history=[]
        self.general_reward_history=[]
        self.total_speaking_reward_history=[]
        self.total_listening_reward_history=[]
        #optimization objects
        self.lambda_val=0.001
        self.value_lr=0.00002
        self.board_lr=0.00002
        self.message_lr=0.00002
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
            
    def state_concat_one(self, full_state):
        reward=full_state[1]
        board_state=full_state[1]
        temp=[]
        for x in board_state:
            for y in x:
                for z in y:
                    temp.append(z)
        board_state=temp
        is_terminal=full_state[2]
        messageOne=full_state[3][0]
        messageThree=full_state[3][2]
        relationVal=[full_state[4][0]]
        x=0
        for y in self.external_message_history:
            if (y==messageOne).all():
                x=1
        if x==0:
            self.external_message_history.append(messageOne)

        msg1_output=self.lstm_msg1_model(tensor_conversion(messageOne,True))
        msg3_output=self.lstm_msg3_model(tensor_conversion(messageThree,True))
        board_output=self.lstm_board_model(tensor_conversion(board_state,True))
        relation_output=self.lstm_relation_model(tensor_conversion(relationVal,True))

        concat_state=tf.concat([msg1_output,msg3_output,board_output,relation_output],1)

        state=self.ffn_model(tf.expand_dims(concat_state,0))

        self.absolute_state_history.append(state)
        return state

    def board_action(self, state):
        (mean,std)=self.board_model(state)

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        action=tf.squeeze(norm_dist.sample(1),axis=0)

        self.orig_action_history.append(action)
        
        action=numpy_conversion(action)
        action=np.nan_to_num(action)

        action=np.insert(action,0,1)

        if action[1]>=5:
            action[1]=5
        if 5>action[1]>=4:
            action[1]=4
        if 4>action[1]>=3:
            action[1]=3
        if 3>action[1]>=2:
            action[1]=2
        if 2>action[1]:
            action[1]=1
        
        for x in range(len(action[2:])):
            if action[x+2]>=2:
                action[x+2]=2
            if 2>action[x+2]>=1:
                action[x+2]=1
            if 1>action[x+2]>=0:
                action[x+2]=0
            if 0>action[x+2]>=-1:
                action[x+2]=-1
            if -2>action[x+2]:
                action[x+2]=-2
                
        board_action=[action[0],action[1],action[2:]]
        self.action_history_self.append(action)
        return board_action, norm_dist
        
    def message_action(self, state):
        (mean,std)=self.message_model(state)

        norm_dist=tf.compat.v1.distributions.Normal(mean,std)
        message3=tf.squeeze(norm_dist.sample(1),axis=0)

        self.orig_message_history.append(message3)

        message3=numpy_conversion(message3)
        
        message_action=message3
        self.message_history_self.append(message3)
        self.msgstate_history.append(state)
        return message_action, norm_dist

    def state_value(self, state):
        value=self.value_model(state)
        return value

    def updateParameters(self,old_absolute_state_1,absolute_state_1,reward,board_action,message_action,normdist_board,normdist_message):
        ####This code regularizes the reward:####
        (board_reward,general_reward,total_speaking_reward,total_listening_reward)=reward
        self.board_reward_history.append(board_reward)
        self.general_reward_history.append(general_reward)
        self.total_speaking_reward_history.append(total_speaking_reward)
        self.total_listening_reward_history.append(total_listening_reward)

        overall_reward=board_reward+general_reward+total_speaking_reward+total_listening_reward
        overall_board_reward=board_reward+general_reward
        overall_message_reward=general_reward+total_speaking_reward+total_listening_reward

        value_error_target=overall_reward+(self.lambda_val*self.state_value(absolute_state_1))
        value_error_overall=(overall_reward+(self.lambda_val*self.state_value(absolute_state_1)))-self.state_value(old_absolute_state_1)
        value_error_board=(overall_board_reward+(self.lambda_val*self.state_value(absolute_state_1)))-self.state_value(old_absolute_state_1)
        value_error_message=(overall_message_reward+(self.lambda_val*self.state_value(absolute_state_1)))-self.state_value(old_absolute_state_1)

        loss_board=-tf.math.reduce_sum(tf.math.log(normdist_board.prob(self.orig_action_history[-1]))*value_error_board)
        loss_message=-tf.math.reduce_sum(tf.math.log(normdist_message.prob(self.orig_message_history[-1]))*value_error_message)
        loss_critic=self.huber_loss(self.state_value(old_absolute_state_1),value_error_target)

        board_grads=tape.gradient(loss_board,self.board_model.trainable_variables)
        message_grads=tape.gradient(loss_message,self.message_model.trainable_variables)
        critic_grads=tape.gradient(loss_critic,self.value_model.trainable_variables)
        ffn_grads=tape.gradient(critic_grads,self.ffn_model.trainable_variables)

        msg1_grads=tape.gradient(ffn_grads,self.lstm_msg1_model.trainable_variables)
        msg3_grads=tape.gradient(ffn_grads,self.lstm_msg3_model.trainable_variables)
        board_lstm_grads=tape.gradient(ffn_grads,self.lstm_board_model.trainable_variables)
        relation_grads=tape.gradient(ffn_grads,self.lstm_relation_model.trainable_variables)

        self.optimizer.apply_gradients(zip(board_grads,self.board_model.trainable_variables))
        self.optimizer.apply_gradients(zip(message_grads,self.message_model.trainable_variables))
        self.optimizer.apply_gradients(zip(critic_grads,self.value_model.trainable_variables))
        self.optimizer.apply_gradients(zip(ffn_grads,self.ffn_model.trainable_variables))
        self.optimizer.apply_gradients(zip(msg1_grads,self.lstm_msg1_model.trainable_variables))
        self.optimizer.apply_gradients(zip(msg3_grads,self.lstm_msg3_model.trainable_variables))
        self.optimizer.apply_gradients(zip(board_lstm_grads,self.lstm_board_model.trainable_variables))
        self.optimizer.apply_gradients(zip(relation_grads,self.lstm_relation_model.trainable_variables))

        agent1_low.append([message_action,board_action,old_absolute_state_1,absolute_state_1,reward[0],reward[1],reward[2],reward[3]])
        
        return overall_reward

def update_plots(reward3,reward1,reward2):
    agent3_reward.append(reward3)
    agent2_reward.append(reward2)
    agent1_reward.append(reward1)
    x=list(range(len(agent3_reward)))
    ax.cla()
    ax.plot(x,agent3_reward,label="agent 3")
    ax.plot(x,agent2_reward,label="agent 2")
    ax.plot(x,agent1_reward,label="agent 1")
    plot.draw()
    plot.pause(0.001)
    

#main program initialization
plot.ion()
plot.show()
tf.keras.backend.set_floatx('float64')
fig=plot.figure(figsize=(12,6), facecolor='#DEDEDE')
ax=plot.subplot(121)
ax.set_facecolor('#DEDEDE')
checkers=checkers_environment()
agent_one=agent_one()
agent_two=agent_two()
agent_three=agent_three()
checkers.env_init()

#make startup variables here
full_state=checkers.env_start()
discount=0.99
terminal=False

episode_history=[]
while True:
    while terminal==False:
        agent3_low=[]
        agent2_low=[]
        agent1_low=[]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(agent_three.lstm_msg1_model.trainable_variables)
            tape.watch(agent_three.lstm_msg2_model.trainable_variables)
            tape.watch(agent_three.lstm_msg3_model.trainable_variables)
            tape.watch(agent_three.lstm_msg4_model.trainable_variables)
            tape.watch(agent_three.lstm_board_model.trainable_variables)
            tape.watch(agent_three.lstm_relation_model.trainable_variables)
            tape.watch(agent_three.ffn_model.trainable_variables)
            tape.watch(agent_three.board_model.trainable_variables)
            tape.watch(agent_three.message_model.trainable_variables)
            tape.watch(agent_three.value_model.trainable_variables)
            
            ###AGENT THREE ROUND###
            #set new states
            absolute_state_3=agent_three.state_concat_three(full_state,0)
            absolute_state_2=agent_two.state_concat_two(full_state)
            absolute_state_1=agent_one.state_concat_one(full_state)
            sfs=full_state
            #get actions
            (board_action,normDistBoard)=agent_three.board_action(absolute_state_3)
            (message_action,normDistMessage)=agent_three.message_action(absolute_state_3)
            #get rewards
            (current_state, is_terminal, board_reward, general_reward)=checkers.env_step_board(board_action)
            (total_speaking_reward, total_listening_reward)=checkers.env_step_message(agent_three.msgstate_history,agent_three.message_history_self,agent_three.action_history_self,agent_three.external_message_history)
            reward=[board_reward,general_reward,total_speaking_reward,total_listening_reward]
            #new state go get relationship vals
            (old_absolute_state_1,old_absolute_state_2,old_absolute_state_3, full_state)=checkers.prep_new_state(message_action,is_terminal,reward,current_state,absolute_state_1,absolute_state_2,absolute_state_3,3)
            absolute_state_3=agent_three.state_concat_three(full_state,0)
            absolute_state_2=agent_two.state_concat_two(full_state)
            absolute_state_1=agent_one.state_concat_one(full_state)
            current_state_value1=agent_one.state_value(absolute_state_1)
            current_state_value2=agent_two.state_value(absolute_state_2)
            old_state_value_1=agent_one.state_value(old_absolute_state_1)
            old_state_value_2=agent_two.state_value(old_absolute_state_2)
            #update parameters and output relations
            (relations,reward3)=agent_three.updateParameters(old_absolute_state_3,absolute_state_3,current_state_value1,current_state_value2,old_state_value_1,old_state_value_2,reward,board_action,message_action,normDistBoard,normDistMessage)
            agent3_low[0].append(sfs)
            print("agent 3 round done")
        
            ###AGENT ONE ROUND###
            #update states to incorporate new relationship values
            full_state=checkers.updateState(relations)
            absolute_state_3=agent_three.state_concat_three(full_state,0)
            absolute_state_2=agent_two.state_concat_two(full_state)
            absolute_state_1=agent_one.state_concat_one(full_state)
            sfs=full_state
            #get actions
            (board_action,normDistBoard)=agent_one.board_action(absolute_state_1)
            (message_action,normDistMessage)=agent_one.message_action(absolute_state_1)
            #get rewards
            (current_state, is_terminal, board_reward, general_reward)=checkers.env_step_board(board_action)
            (total_speaking_reward, total_listening_reward)=checkers.env_step_message(agent_one.msgstate_history,agent_one.message_history_self,agent_one.action_history_self,agent_one.external_message_history)
            reward=[board_reward,general_reward,total_speaking_reward,total_listening_reward]
            #new full_state
            (old_absolute_state_1,old_absolute_state_2,old_absolute_state_3, full_state)=checkers.prep_new_state(message_action,is_terminal,reward,current_state,absolute_state_1,absolute_state_2,absolute_state_3,1)
            #update parameters
            reward1=agent_one.updateParameters(old_absolute_state_1,absolute_state_1,reward,board_action,message_action,normDistBoard,normDistMessage)
            agent1_low[0].append(sfs)
            print("agent 1 round done")

            ###AGENT TWO ROUND###
            #update states to account for agent 1's action
            absolute_state_3=agent_three.state_concat_three(full_state,1)
            absolute_state_2=agent_two.state_concat_two(full_state)
            absolute_state_1=agent_one.state_concat_one(full_state)
            sfs=full_state
            #get actions
            (board_action,normDistBoard)=agent_two.board_action(absolute_state_2)
            (message_action,normDistMessage)=agent_two.message_action(absolute_state_2)
            #get rewards
            (current_state, is_terminal, board_reward, general_reward)=checkers.env_step_board(board_action)
            (total_speaking_reward, total_listening_reward)=checkers.env_step_message(agent_two.msgstate_history,agent_two.message_history_self,agent_two.action_history_self,agent_two.external_message_history)
            reward=[board_reward,general_reward,total_speaking_reward,total_listening_reward]
            #new full_state
            (old_absolute_state_1,old_absolute_state_2,old_absolute_state_3, full_state)=checkers.prep_new_state(message_action,is_terminal,reward,current_state,absolute_state_1,absolute_state_2,absolute_state_3,2)
            #update parameters
            reward2=agent_two.updateParameters(old_absolute_state_2,absolute_state_2,reward,board_action,message_action,normDistBoard,normDistMessage)
            agent2_low[0].append(sfs)
            print("agent 2 round done")
            tape.reset()
            update_plots(reward3,reward1,reward2)
            agent3.append(agent3_low[0])
            agent2.append(agent2_low[0])
            agent1.append(agent1_low[0])
        if is_terminal==True:
            terminal=True
    episode_history.append("placeholder")
    full_state=checkers.env_start()
    terminal=False
