import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dfx=pd.read_csv("Linear_X_Train.csv")
x_train=dfx.values

dfy=pd.read_csv("Linear_Y_Train.csv")f
y_train=dfy.values

dfz=pd.read_csv("Linear_X_Test.csv")
x_test=dfz.values

x_train=(x_train-x_train.mean())/(x_train.std())
plt.scatter(x_train,y_train) #giving a oval shape and increasing
plt.show()



def hypo(x,theta):
    #y= thetha[0]+x*thetha[1]
    #return y
    return theta[0] + theta[1] * x

def error(X,Y,theta):
    m=X.shape[0]
    error=0
    for i in range(m):
        hx=hypo(X[i],theta)
        error+=(hx-Y[i])**2
    return error


def gradient(X,Y,theta):
    m=X.shape[0]
    grad=np.zeros((2,))
    for i in range(m):
        hx=hypo(X[i],theta)
        grad[0]+=(hx-Y[i])
        grad[1]+=(hx-Y[i]) * X[i]
    grad[0]=grad[0]/m
    grad[1]=grad[1]/m

    return grad



def gradient_descent(X,Y,learning_rate=0.1,max_steps=100):
    theta = np.zeros((2, ))
    error_list=[]
    theta_list=[]
    for i in range(max_steps):
        grad=gradient(X,Y,theta)
        e=error(X,Y,theta)
        error_list.append(e)
        theta[0]=theta[0] - (grad[0] * learning_rate)
        theta[1]=theta[1] - (grad[1] * learning_rate)
        theta_list.append((theta[0],theta[1]))

    return theta,error_list,theta_list

f_theta,f_elist,f_tlist=gradient_descent(x_train,y_train)
f_tlist=np.array(f_tlist)
y_test=hypo(x_test,f_theta)

#plt.plot(f_tlist[:,0],color="orange")
#plt.plot(f_tlist[:,1],color="green")
#plt.ion()

#plt.scatter(x_train,y_train,label="Training Data")
#plt.plot(x_train,y_test,color="red",label="pred")

#plt.legend()
#plt.show()
#plt.draw()
#plt.pause(20)
#plt.clf()
#plt.show()


df=pd.DataFrame(data=y_test,columns=["y"])

df.to_csv("Hardwork_y_test.csv",index=False)