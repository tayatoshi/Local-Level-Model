
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd


# In[ ]:

alpha=[]
alpha.append(np.exp(2)+80)
for i in range(50):
    alpha.append(alpha[i]+np.random.normal(0,50))
y=alpha+np.random.normal(0,50,len(alpha))

plt.plot(y,label="observation")
plt.plot(alpha,label="state")
plt.legend()
plt.ylim(min(y)-5,max(y)+5)
plt.show()


# In[ ]:

def kalman(a0=np.mean(alpha),p0=5000,Var_epsilon=50,Var_eta=50):
    a = a0
    p = p0
    A = []
    A_t = []
    P_t = []
    for i in range(50):
        u = y[i] - a
        f = p + Var_epsilon
        K = p/f
        L = 1-K
        a_t = a + K*u
        A_t.append(a_t)
        p_t = p*L
        P_t.append(p_t)
        A.append(a)
        a = A_t[i]
        p = P_t[i] + Var_eta
        
    A_t=pd.DataFrame(A_t)
    A_t.index=A_t.index+1
    plt.plot(y,'-o',label='y')
    plt.plot(alpha,label="state")
    #plt.plot(A,label="one step")
    plt.plot(A_t,label="filtering state")
    #plt.plot(A_t-1.96*np.sqrt(P_t))
    #plt.plot(A_t+1.96*np.sqrt(P_t))
    #plt.legend()
    plt.ylim(min(y)-50,max(y)+50)
    plt.show()


# In[ ]:

kalman()

