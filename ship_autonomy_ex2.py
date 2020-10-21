import numpy as np
import matplotlib.pyplot as plt


M = [[1e7, 0, 0],[0,1e7, 8.4e6], [0,8.4e6, 5.8e9]]
M = np.array(M).reshape(3,3)

D = [[3e5, 0, 0], [0, 5.5e5, 6e5], [0, 6e5, 1.3e8]]
D = np.array(D).reshape(3,3)

t_end = 1000
h = 1
inc = int(np.round(t_end/h))
time = np.linspace(0,t_end, inc+1)
T = 800
r = 50

delta_end = np.zeros((1,len(time)))
delta_change = 3.7
for n in range(inc):


    delta_end[:,n + 1] = delta_end[:,n] + delta_change
    if delta_end[:,n] >= 20:
        delta_change = -3.7
    elif delta_end[:,n] <= -20:
        delta_change = 3.7


delta = delta_end


X = T*np.cos(np.deg2rad(delta))
Y = T*np.sin(np.deg2rad(delta))
N = Y*r



torque = [X, Y, N]
torque = np.array(torque).reshape(3,len(time))


nu = np.zeros((len(D), len(time)))
eta = np.zeros((len(D), len(time)))



for i in range(inc):
    f = np.dot(np.linalg.inv(M), (torque[:,i] - np.dot(D,nu[:,i])))

    nu[:,i + 1] = f*(time[1] - time[0]) + nu[:,i]

    phi = eta[2,i]
    R = [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    R = np.array(R).reshape(3, 3)

    eta1 = np.dot(R,nu[:,i])

    eta[:,i + 1] = eta1*(time[1] - time[0]) + eta[:,i]

plt.plot(eta[0,:], eta[1,:])
plt.show()
