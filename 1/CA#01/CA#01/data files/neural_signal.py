
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


from neural_signal_data import y, s, T, N, M, a_true
from neural_signal_data import visualize_data
from neural_signal_data import visualize_estimate
from neural_signal_data import visualize_polished
from neural_signal_data import find_nonzero_entries
# Let us start by visualizing the observed data
visualize_data()
plt.savefig('observed_signal.pdf')


# Solve part b)
a = cp.Variable(N)
lambda_reg = 2
objective = cp.Minimize(cp.sum_squares(cp.conv(s,a).flatten() - y)/(T) \
	+ lambda_reg * cp.sum(a))
constraints = [0 <= a]
prob = cp.Problem(objective, constraints)
result = prob.solve()


# Plot part b)
visualize_estimate(a.value)
plt.savefig('Reconstruction_with_regularization.pdf')

print('Non-zeros of true a:',find_nonzero_entries(a_true))
print('Non-zeros of estimated $\hat{a}$:',find_nonzero_entries(a.value))


# Solve part c)
ind = np.where(a.value <=0.001)[0]
a_p = cp.Variable(N)
objective = cp.Minimize(cp.sum_squares(cp.conv(s,a_p).flatten() - y)/(T) )
constraints = [0 <= a_p, a_p[ind] == 0]
prob = cp.Problem(objective, constraints)
result = prob.solve()

# Plot part c)
visualize_polished(a_p.value)
plt.savefig('Reconstruction_with_polishing.pdf')



#%% Alternative solution to part b)


#a = cp.Variable(N)
#lambda_reg = 2
#objective = cp.Minimize(cp.square(cp.norm(cp.conv(s,a).flatten() - y,2))/(T) \
#	+ lambda_reg * cp.sum(a))
#constraints = [0 <= a]
#prob = cp.Problem(objective, constraints)
#result = prob.solve()

#visualize_estimate(a.value)
#print('Non-zeros of true a:',find_nonzero_entries(a_true))
#print('Non-zeros of estimated $\hat{a}$:',find_nonzero_entries(a.value))

# Altenation solution to part c)


#ind = np.where(a.value <=0.001)[0]
#a_p = cp.Variable(N)
#objective = cp.Minimize(cp.square(cp.norm(cp.conv(s,a_p).flatten() - y,2))/(T) )
#constraints = [0 <= a_p, a_p[ind] == 0]
#prob = cp.Problem(objective, constraints)
#result = prob.solve()

#visualize_polished(a_p.value)
#plt.savefig('Reconstruction_with_polishing.pdf')


#%% Enhanced polishing

'''
Traditionally, the enhanced polishing would have been automated, but here 
the problem is simple enough we can prune the close non-zero activations by hand.

We start with the non-zeros of estimated \hat a from part b): [198  242  249  
                345  459  499  799  849 1098 1499 1698]

Here, only 242 and 249 are closer to each other than 10 frames, e.g. 1ms. We
prune this activation by taking the middle point, 246. Then, we solve the
optimization problem again, this time with the pruned non-sparse activation
pattern.
'''

ind_nonzero = np.array([ 198,  247,  345,  459,  499,  799,  849, 
                        1098, 1499, 1698])


# Solve part c)
ind = np.setdiff1d(np.linspace(0,N-1,N).astype(int),ind_nonzero)
a_ep = cp.Variable(N)
objective = cp.Minimize(cp.sum_squares(cp.conv(s,a_ep).flatten() - y)/(T) )
constraints = [0 <= a_p, a_ep[ind] == 0]
prob = cp.Problem(objective, constraints)
result = prob.solve()

# Plot part c)
from neural_signal_data import times
plt.subplots(2,1)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

plt.subplot(311)
plt.plot(times,y)
plt.ylabel('y (mV)')

plt.subplot(312)
plt.plot(times,np.convolve(s,a_true),label = 'True',linewidth = 3)
plt.plot(times,np.convolve(s,a_ep.value),label = 'Estimated',linestyle = '-.')
plt.ylabel(r's * $\hat{a}_{enhanced + polished}$')

plt.subplot(313)
plt.plot(times[:2000],a_true,label = '$True$',linewidth = 3)
plt.plot(times[:2000],a_ep.value,label = '$Estimated$',linestyle = '-.')
plt.ylabel(r'$\hat{a}_{enhanced + polished}$')
plt.legend()
plt.xlabel('t')
plt.savefig('Reconstruction_with_enhanced_polishing.pdf')




