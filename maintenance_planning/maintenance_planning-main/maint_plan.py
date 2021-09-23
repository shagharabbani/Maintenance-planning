

from os import stat
from typing import AsyncGenerator
import numpy as np
from numpy.core.defchararray import array
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
from utils import *
import copy, time, os
import tensorflow as tf
from policynet import CNNPolicy

np.random.seed(0)

#Initialize start time
start=time.time()


batch_size = 1000
startE = 1.0; endE = 0.01; anneling_steps = int(1e2); e_step = (startE-endE)/anneling_steps; e = startE


noc=5       # number of components
nosp=6      # number of sub-periods [weeks]
nos=4       # number of states 
noph=26     # number of planning horizons [each ph: 6 weeks]
max_age=3   # maximum age of each component [weeks]
tt=noph*nosp    # total time [weeks]
noe=50000     # number of episodes
pre_train_steps=10000
rnd = np.random.rand()
DQNloss=[0]
costss = [0]

myBuffer = experience_buffer(pre_train_steps)

tf.reset_default_graph()
tf.disable_eager_execution()




mH = 128
mainQN = CNNPolicy(); mainQN.create_network(mH)
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1000)

sess = tf.Session()
sess.run(init)

components=range(noc)
Time=range(tt)


episodeBuffer = experience_buffer(pre_train_steps)




Q_dict = {}
Q_all=[]

for episode in range(noe+1):

    # print("@@@ EPISODE: "+str(episode)+" @@@")

    # preallocations for beginning of each episode
    ages=np.zeros([tt,noc])
    TTFs=np.zeros([tt,noc])
    states_int=np.zeros([tt,noc],dtype=int)
    states_mat=np.zeros([tt,noc*(nos+1)],dtype=int)
    actions=np.zeros([tt,noc],dtype=int)
    actions_mat=np.zeros([tt,noc*2],dtype=int)
    is_unplanneds=np.zeros([tt,noc],dtype=int)
    costs=np.zeros([tt,noc])
    rewards=np.zeros([tt,noc])
    V=np.zeros([tt,noc])
    inventories=np.zeros([tt],dtype=int)
    productions=np.zeros([tt],dtype=int)

    demands=np.zeros([tt],dtype=int)

    age=np.zeros(noc,dtype=int)
    #TTF formula
    TTF=np.zeros(noc,dtype=int)
    alpha=np.zeros(noc)
    beta=np.zeros(noc)
    demand=np.zeros(nosp)
    is_unplanned=np.zeros(noc)
    
    init_inventory=10
    init_production=5

    for t in Time:
        # print("* t = "+str(t)+" *")
        # Generate random initial age for noc components.
        # age=np.random.randint(1,max_age,noc)
        # inventory=inventories[0]
        if t%6==0:
            demand=np.random.randint(0,10,nosp)
        d=demand[t%6]
        demands[t]=d

        if t==0:
            
            # age[0]=2; age[1]=1; age[2]=5; age[3]=3; age[4]=3
            age=np.random.randint(5,15,noc)
            # print("age: "+str(age))

            beta=np.random.uniform(1.4,2,noc)       # no units
            alpha=np.random.randint(40,200,noc)     # [weeks]

            
            for c in components:
                TTF[c]=int(alpha[c]*(np.power(-np.log(1-np.random.rand()),(1/beta[c]))))
                # TTF[c]=int(alpha[c]*np.log(-np.power(1-np.random.rand(),(1/beta[c]))))
                if (TTF[c]<age[c]):
                    TTF[c]+=age[c]      ### TO BE REVISED LATER ###
                
            # TTF[0]=7; TTF[1]=4;TTF[2]=8; TTF[3]=6; TTF[4]=10

            # generate next TTF
            # Shape Parameter,Scale Parameter
            

            inventories[0]=init_inventory
            # inv=num_decoder(inventories[0],2)
            # q=0
            productions[0]=init_production

        else:
            
            for c in components:
                if actions[t-1][c]==1:
                    # generate next age
                    age[c]=1
                    # generate next TTF
                    # Shape Parameter,Scale Parameter
                    beta[c]=np.random.uniform(1.4,2,1)       # no units
                    alpha[c]=np.random.randint(40,200,1)     # [weeks]
                    TTF[c]=age[c]+int(alpha[c]*(np.power(-np.log(np.random.rand(1)),(1/beta[c]))))

                else:
                    # generate next age
                    age[c]+=1
                    alpha[c]=alpha_p[c]
                    beta[c]=beta_p[c]
                    TTF[c]=TTFs[t-1][c]

            inventories[t]=inventories[t-1]+productions[t-1]-demands[t-1]

            



        


        # print("age: "+str(age))
        # print("TTF: "+str(TTF))

                
        # Conditional_Failure_prob(components,age,alpha,beta)


        # state=np.zeros(noc,dtype=int)
        # state_next=np.zeros(noc,dtype=int)

        # State of each component in one planning horizon (i.e. nosp weeks)
        # cond_prob=np.ones([noc,nosp])
        # for c in components:
        #     # print("cond_prob: "+str(cond_prob))
        #     # print("CFp: "+str(Conditional_Failure_prob(c,age[c],alpha[c],beta[c],TTF[c])))
        #     cond_prob[c,:]=Conditional_Failure_prob(c,age[c],alpha[c],beta[c],TTF[c],nosp)

        cond_prob=Conditional_Failure_prob(components,age,alpha,beta)



        # print("cond_prob: "+str(cond_prob))
        # print(int_state(cond_prob))
        # state_int=np.zeros(noc,dtype=int)

        
        

        state_mat=np.zeros([noc,nos+1],dtype=int)

        # inv=num_encoder(ini)
        state_int=int_state(cond_prob)

        state_mat[:,0:4]=processsa(state_int,nos)
        
        state_mat[:,4]=num_decoder(inventories[t]-demands[t]+16,2)
        

        # print("state_int: "+str(state_int))


        is_unplanned=maint_planning(age,TTF)

        is_maintenance=np.zeros(noc)

        if rnd < e:
            # Generating a random action for each 7 component (1x7 vector, each entry is 0, 1, 2, or 3).
            a = np.random.randint(0,2,noc)
        else:
            a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[np.reshape(state_mat,25)]})[0]


        # Explore vs exploit
        if t%10==1:
            a = np.random.randint(0,2,noc)



        is_maintenance=a
        # is_maintenance=action_maint1d(state_int,is_unplanned)
        # is_maint_mat=action_mat1d(state_mat,is_unplanned)
        is_maint_mat=processsa(is_maintenance,2)


        # print("action: "+str(is_maintenance))
        
        # print("is_unplanned: "+str(is_unplanned))
        # state=np.array([0,1])


        inv=inventories[t]

        # Finding the production level  based on the chosen action
        if sum(a[:])==0:
            q=1
        else:
            q=0

        if inventories[t]>=demands[t]:
            # productions[t]=q*demands[t]
            productions[t]=0
        else:
            if inventories[t]>0:
                productions[t]=q*(d-inventories[t-1])
            else:
                productions[t]=q*(d-inventories[t-1])


        cost=cost_func(components,state_int,is_maintenance,is_unplanned,d,inv)
        # print("cost: "+str(cost))
        reward=-cost/400000



        ages[t,:]=copy.deepcopy(age)
        alpha_p=copy.deepcopy(alpha)
        beta_p=copy.deepcopy(beta)
        TTFs[t,:]=copy.deepcopy(TTF)
        states_int[t,:]=copy.deepcopy(state_int)
        states_mat[t,:]=copy.deepcopy(np.reshape(state_mat,25))
        actions[t,:]=copy.deepcopy(is_maintenance)
        actions_mat[t,:]=copy.deepcopy(np.reshape(is_maint_mat,10))
        is_unplanneds[t,:]=copy.deepcopy(is_unplanned)
        rewards[t]=copy.deepcopy(reward)
    V=getreturn(rewards); # can be interpreted as Value function
    costss.append(np.sum(rewards))

    if episode%100==0:
        print(str(episode), '***********************', costss[-1],  DQNloss[-1], '***********************', time.time() - start)
        



    # print("###############################################")

    for j in Time:
        s=states_mat[j,:]; a=actions[j,:]; Q=V[j]; r=rewards[j];q=states_mat[j,21:]
        # print("Q["+str(j)+"]: "+str(Q)) 
        if j<tt-1:
            s1=states_int[j+1,:]
        else:
            s1=np.zeros(noc*nos) # TO BE REVISED LATER

        # _,state_num=np.where(s==1)

        for c in range(noc):
            tuple_idx=(c,s[c],a[c])

            if Q_dict.get(tuple_idx):
                n_times,Q_value=Q_dict[tuple_idx]
                n_times+=1
                Q_value+=(Q[c]-Q_value)*1
                Q_dict[tuple_idx]=(n_times,Q_value)
            else:
                n_times=0
                Q_value=Q[c]
                Q_dict[tuple_idx]=(n_times,Q_value)
            Q[c]=Q_value

        # svec=np.reshape
        # episodeBuffer.add(np.array([np.array([s],dtype=object),np.array([a],dtype=object),np.array([Q],dtype=object),np.array([r],dtype=object),np.array([s1],dtype=object)],dtype=object))
        z=np.array([np.array([0,0]),np.array([0]),np.array([0]),np.array([0]),np.array([0])],dtype=object)
        z[0]=s; z[1]=a; z[2]=Q; z[3]=r; z[4]=s1;
        episodeBuffer.add(np.reshape(z,[1,5]))
    myBuffer.add(episodeBuffer.buffer)


    if episode >100:
        if e>endE:
            e -= e_step
        trainBatch = myBuffer.sample(batch_size)
        input_s = np.vstack(trainBatch[:,0])
        input_a = np.vstack(trainBatch[:,1])
        target_Q = np.vstack(trainBatch[:,2])

        _, qloss = sess.run([mainQN.updateModel,mainQN.loss],
                 feed_dict={mainQN.scalarInput:input_s, mainQN.actions:input_a, mainQN.targetQ:target_Q})
        DQNloss.append(qloss)


    if episode%5000 == 0:
        print(rnd>e)
        filepath = 'result/training results/step' + str(noe)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        saver.save(sess, filepath+'/training-'+str(episode)+'.cpkt')
        np.save(filepath+'/costs.npy', costss)
        np.save(filepath+'/DQNloss.npy', DQNloss)
        np.save(filepath+'/Q_dict.npt',Q_dict)
        print("Save Model")
        elapsed = time.time()-start
        print(episode,e,elapsed,costss[-1])
        state_int = randomint()
        print(state_int)
        s = np.reshape(state_int,[25])
        print(sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0])
        start = time.time()

    # Q_all.append(Q_dict)


print("Check")


# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print("AGES: "+str(ages))
# print("TTFS: "+str(TTFs))
# print("STATES: "+str(states_int))
# print("ACTIONS: "+str(actions))
# print("IS_PLANNED: "+str(is_unplanneds))
# print("V: "+str(V))
# print("Total cost: "+str(sum(rewards)))
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")