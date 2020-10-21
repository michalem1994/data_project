import numpy as np
import matplotlib.pyplot as plt




M = [[1e7, 0, 0],[0,1e7, 8.4e6], [0,8.4e6, 5.8e9]]
M = np.array(M).reshape(3,3)

D = [[3e5, 0, 0], [0, 5.5e5, 6e5], [0, 6e5, 1.3e8]]
D = np.array(D).reshape(3,3)


force = np.transpose([1e5, 0, 1e5])


tend = 2000
h = 1
N = int(np.round(tend/h))
t = np.linspace(0,tend,N+1)

v = np.zeros((len(D), len(t)))
eta = np.zeros((len(D), len(t)))




for i in range(N):
    f = np.dot(np.linalg.inv(M),(force - np.dot(D,v[:,i])))
    v[:,i + 1] = f*(t[1] - t[0]) + v[:,i]

    phi = eta[2,i]
    R = [[-np.sin(phi), np.cos(phi), 0], [np.cos(phi), np.sin(phi), 0], [0, 0, 1]]
    R = np.array(R).reshape(3, 3)

    eta1 = np.dot(R,v[:,i])

    eta[:,i+1] = eta1*(t[1] - t[0]) + eta[:,i]


fig, ax = plt.subplots(3, 1)
ax[0].plot(t[:],v[0,:])
ax[0].set_title('u')
ax[0].set_ylim([0,0.4])
ax[0].set_xlim([0,t[-1]])

ax[1].plot(t[:],v[1,:])
ax[1].set_title('v')
ax[1].set_xlim([0,t[-1]])

ax[2].plot(t[:],v[2,:]*180/np.pi)
ax[2].set_title('phi')
ax[2].set_xlim([0,t[-1]])

fig1, ax2 = plt.subplots(2,1)
ax2[0].plot(eta[1,:], eta[0,:])
ax2[0].set_title('x')
ax2[0].set(xlabel='x', ylabel='y')


ax2[1].plot(t, eta[2,:]*180/np.pi)
ax2[1].set_title('heading')
ax2[1].set(xlabel='time', ylabel='heading')
ax2[1].set_xlim([0,t[-1]])

plt.show()
