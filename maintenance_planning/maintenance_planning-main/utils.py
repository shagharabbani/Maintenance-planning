
import numpy as np
import random

def Failure_CDF(c,t,alpha,beta):
    F=1-np.exp(-(t/alpha)**beta)
    # print("Failure CDF: "+str(F))
    return F

def Conditional_Failure_prob(c,age,alpha,beta):
    # print("age: "+str(age))
    # print("alpha: "+str(alpha))
    # print("beta: "+str(beta))
    # print("TTF: "+str(TTF))
    # conditionalprob=np.ones(nosp)
    # print("TTF-age: "+str(nosp))
    # for tau in range(1,TTF-age+1):
        # conditionalprob[tau-1]=(Failure_CDF(c,age+tau,alpha,beta)-Failure_CDF(c,age,alpha,beta))/(1-Failure_CDF(c,age,alpha,beta))
    # print("Conditional prob.: "+str(conditionalprob))
    comp=len(c)
    conditionalprob=np.zeros(comp)
    for i in range(comp):
        F1=Failure_CDF(c[i],age[i]+1,alpha[i],beta[i])
        F2=Failure_CDF(c[i],age[i],alpha[i],beta[i])
        conditionalprob[i]=(F1-F2)/(1-F2)

    return conditionalprob

def num_encoder(inputs,base):
    num = 0
    assert inputs.shape == (5,)
    for i in range(inputs.size):
        num += inputs[i]*base**(4-i)
    return num

def num_decoder(inputs,base):
    outs = np.zeros(5)
    for i in range(5):
        outs[i] = int(divmod(inputs,base**(4-i))[0])
        inputs -= outs[i]*base**(4-i)
    return np.int32(outs)


def int_state(float_state):
    # I,J=float_state.shape
    # state=-np.ones([I,J],dtype=int)
    # for i in range(I):
    #     for j in range(J):
    #         if (float_state[i][j]>=0 and float_state[i][j]<0.4):
    #             state[i][j]=0
    #         elif (float_state[i][j]>=0.4 and float_state[i][j]<0.6):
    #             state[i][j]=1
    #         elif (float_state[i][j]>=0.6 and float_state[i][j]<1):
    #             state[i][j]=2
    #         elif (float_state[i][j]==1):
    #             state[i][j]=3
    #         else:
    #             print("### SOMETHING IS WRONG! ###")

    # return state

    I=float_state.shape[0]
    state=np.zeros(I,dtype=int)
    for i in range(I):
        if (float_state[i]>=0 and float_state[i]<0.4):
            state[i]=0
        elif (float_state[i]>=0.4 and float_state[i]<0.6):
            state[i]=1
        elif (float_state[i]>=0.6 and float_state[i]<0.9):
            state[i]=2
        elif (float_state[i]>=0.9 and float_state[i]<1):
            state[i]=3
        else:
            state[i]=3
            # print("### SOMETHING IS WRONG! ###")
            # print("i: "+str(i))
            # print("float_state: "+str(float_state))
    return state


def mat_state(float_state):
    # I,J=float_state.shape
    # state=-np.ones([I,J],dtype=int)
    # for i in range(I):
    #     for j in range(J):
    #         if (float_state[i][j]>=0 and float_state[i][j]<0.4):
    #             state[i][j]=0
    #         elif (float_state[i][j]>=0.4 and float_state[i][j]<0.6):
    #             state[i][j]=1
    #         elif (float_state[i][j]>=0.6 and float_state[i][j]<1):
    #             state[i][j]=2
    #         elif (float_state[i][j]==1):
    #             state[i][j]=3
    #         else:
    #             print("### SOMETHING IS WRONG! ###")

    # return state

    I=float_state.shape[0]
    J=4
    state=np.zeros([I,J],dtype=int)
    for i in range(I):
        if (float_state[i]>=0 and float_state[i]<0.4):
            state[i][0]=1
        elif (float_state[i]>=0.4 and float_state[i]<0.6):
            state[i][1]=1
        elif (float_state[i]>=0.6 and float_state[i]<0.9):
            state[i][2]=1
        elif (float_state[i]>=0.9 and float_state[i]<1):
            state[i][3]=1
        else:
            print("### SOMETHING IS WRONG! ###")
            print("i: "+str(i))
            print("float_state: "+str(float_state))
    return state

def do_maintenance_prob(from_ind,to_ind):
    maint_mat=np.array([[0.05, 0.25, 0.75, 0.96],
    [0, 0.3, 0.85, 0.97],[0,0,0.65,0.98],[0,0,0,1]])

    return maint_mat[from_ind][to_ind]

def do_maintenance_prob1d(ind):
    """
    docstring
    """
    maint_mat=np.array([0.45, 0.75, 0.95, 0.98])
    return maint_mat[ind]

# states_int=np.array([[2,3,3,3,3,3],[1,2,2,2,3,3]])

def action_maint(states_int):
    comp=states_int.shape[0]
    action=np.zeros(comp,dtype=int)
    for c in range(comp):
        state_now=states_int[c][0]
        state_next=states_int[c][1]
        act_prob=do_maintenance_prob(state_now,state_next)
        # 0: postpone, 1: do maintenance
        action[c]=np.random.choice(2, 1, p=[1-act_prob, act_prob])

    return action




def action_maint1d(states_int,planning):
    comp=states_int.shape[0]
    action=np.zeros(comp,dtype=int)
    for c in range(comp):
        if planning[c]==1:
            action[c]=1
        else:
            state_now=states_int[c]
            # state_next=states_int[c][1]
            act_prob=do_maintenance_prob1d(state_now)
            # 0: postpone, 1: do maintenance
            action[c]=np.random.choice(2, 1, p=[1-act_prob, act_prob])

    return action

def action_mat1d(states_int,planning):
    comp=states_int.shape[0]
    action=np.zeros([comp,2],dtype=int)
    for c in range(comp):
        if planning[c]==1:
            action[c][1]=1
        else:
            state_now=states_int[c]
            # state_next=states_int[c][1]
            act_prob=do_maintenance_prob1d(state_now)
            # 0: postpone, 1: do maintenance
            temp=np.random.choice(2, 1, p=[1-act_prob, act_prob])
            if temp==0:
                action[c][0]=1
            else:
                action[c][1]=1


    return action

# print(do_maintenance_prob(1,2))
# print(action_maint1d(states_int))
# action_vec=action_maint1d(states_int)

def processsa(input,base):
    # 6 for states,  4 for actions
    num = input.size
    out = np.zeros([num,base],dtype=np.int32)
    for i,a in enumerate(input):
        out[i,:] = np.eye(1,base,a)
    return out


def maint_planning(age,TTF):
    comp=age.shape[0]
    output=np.zeros(comp,dtype=int)
    for c in range(comp):
        # if (action[c]==0):  # postpone
        #     output[c]=0     # planned
        # else:   # do maintenance
        #     if TTF[c]>age[c]:
        #         output[c]=0     # planned
        #     else:
        #         output[c]=1     # unplanned
        if TTF[c]>age[c]:
            output[c]=0     # planned
        else:
            output[c]=1     # unplanned

    return output




def cost_func(components,states_int,is_maintenance,is_unplanned,demand,inv):

    
    risk=np.array([[0.08,0.085,0.09,0.1],[0.72,0.765,0.81,0.9]])
    maint_cost=np.array([[5000,10000],[7000,20000],[4000,15000],[8000,25000],[6000,27000]])
    ch=15 # holding cost per item
    cb=250  # backorder cost per item
    cost=np.zeros(len(components))
    for c in components:
        # print("c: "+str(c))
        # print("is_unplanned[c]: "+str(is_unplanned[c]))
        # print("main_cost[c][is_planned[c]]: "+str(maint_cost[c][is_unplanned[c]]))
        # print("is_maintenance[c]: "+str(is_maintenance[c]))
        # print("states_int[c]: "+str(states_int[c]))
        # print("risk[is_maintenance[c]][states_int[c]]: "+str(risk[is_maintenance[c]][states_int[c]]))
        if is_unplanned[c]==0:
            cost[c]+=maint_cost[c][0]*risk[is_maintenance[c]][states_int[c]]
        else:
            cost[c]+=maint_cost[c][1]

        
        # if sum(is_maintenance)==0:
        #     q=1
        # else:
        #     q=0
            

        # next_inv=inv+q-demand
        if inv>=0:
            cost[c]+=ch*(inv)/len(components)
        else:
            cost[c]+=cb*(-inv)/len(components)


        
        


        # print(cost)


    return cost


def getreturn(reward_list):  # calculate return of a trajectory
    # gamma is the discount factor
    a,b = reward_list.shape
    G = np.zeros([a,b])
    for i in range(a):
        for j in range(i,a):
            G[i] += 1**(j-i)*reward_list[j,:]
    return G

class experience_buffer():
    def __init__(self, buffer_size):  # buffer size = 50000
        self.buffer = []
        self.buffer_size = buffer_size
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, len(self.buffer[0])])


def randomint():
        s = np.random.randint(0,4,5)
        q = np.random.randint(0,32,1)[0]
        raw_s = np.zeros([5,5],dtype=np.int32)
        raw_s[:,0:4] = processsa(s,4)
        raw_s[:, 4] = q_encoder(q)
        # if stau == True:
        #     plot_state(raw_s,7,7)
        state=raw_s
        # self.state_num = s
        return state


def q_encoder(q):
    coder = np.zeros(5,dtype=np.int32)
    for i in range(5):
        coder[i] = divmod(q,2**(4-i))[0]
        q -= coder[i]*2**(4-i)
    return coder