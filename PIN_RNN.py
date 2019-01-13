%reset
import math
import numpy as np
import holoviews as hv
hv.extension('bokeh')
import matplotlib.pyplot as plt
import pandas as pd
from numpy import random as rd

# Initialize network's parameters
N = 500
g = 2
dt = 0.01
time = np.arange(0,100,dt)
tau = 0.1
P0 = 1

# Target functions
targets = np.zeros([N, len(time)])
for i in range(N):
    targets[i,:] = (np.sin(time + 10*rd.random(1))+
                    np.sin(0.4*time + 10*rd.random(1)) +
                    np.sin(0.8*time + 10*rd.random(1)) +
                    np.sin(1.2*time + 10*rd.random(1)) +
                    np.sin(1.6*time + 10*rd.random(1)))
# Plots network currents
plt.subplots(figsize = (17,5))
plt.subplot(2,1,1)
plt.imshow(np.tanh(targets), cmap = "pink")
plt.xlabel("Time (ms)")
plt.ylabel("Rates")
plt.colorbar()
plt.title("Target")
plt.subplot(2,1,2)
plt.imshow(targets, cmap = "pink")
plt.xlabel("Time (ms)")
plt.ylabel("Current")
plt.colorbar()
plt.show()
# Plots sample target neurons' rates
plt.figure(figsize = (15,5))
for i in np.arange(0,5):
    plt.subplot(5,1,i+1)
    plt.plot(np.tanh(targets[i,:]), linewidth = 1)
    plt.ylabel("Rate")
    plt.xlabel("Time (ms)")

# ------------------------------------------------------------------------------

# Train network
ij = g * rd.randn(N, N) / math.sqrt(N)
ij0 = ij.copy()
ijs = ij
errors = 0
PJ = P0 * np.eye(N,N) # Initialization of P0 for training
for runs in np.arange(0,20):
    Rates = np.zeros([N,len(time)])
    current = targets[:,0]
    run_error = 0
    for t in np.arange(0,len(time)):
        Rates[:,t] = np.tanh(current) # Add rate to traces
        weighted = ij.dot(Rates[:,t])
        current = (-current + weighted)*dt/tau + current # Update rates
        # Training
        if t % 10 == 0:
            err = weighted - targets[:,t] # e(t) = z(t) - f(t)
            run_error = run_error + np.mean(err ** 2)
            r_slice = Rates[:, t].reshape(N, 1) # Rates of learning neurons at time t
            k = PJ.dot(r_slice)
            rPr = r_slice.T.dot(k)[0, 0]
            c = 1.0/(1.0 + rPr)
            PJ = PJ - c*(k.dot(k.T)) # P(t) = P(t-1) - ...
            #ijs = np.dstack([ijs, ij]) # Save IJ Matrix before updating
            ij = (ij - (c * np.outer(err.flatten(), k.flatten())))
    errors = np.append(errors, run_error)
    print(runs)


# ------------------------------------------------------------------------------

# Plots error across Training
plt.figure(figsize = (10,5))
plt.scatter(x = np.arange(0,len(errors)), y = errors, s = 10)
plt.plot(errors, c = "Salmon", linewidth = 1.2)
plt.title("Error across training")
plt.show()

# Plot IJ Matrices before and after training
plt.figure(figsize = (20,5))
plt.subplot(1,3,1)
plt.imshow(ij0, cmap = "pink")
plt.title("First IJ")
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(ij, cmap = "pink")
plt.title("Last IJ")
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(np.subtract(ij,ij0), cmap = "BrBG")
plt.title("Difference")
plt.colorbar()
plt.show()

# Plot IJ Matrices before and after training
plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
plt.imshow(ij0, cmap = "pink")
plt.colorbar()
plt.title("Pre-Training IJ")
plt.subplot(2,2,2)
plt.scatter(x = np.linalg.eigvals(ij0).real, y = np.linalg.eigvals(ij0).imag, c = "Salmon", s = 20)
plt.axvline(x = 1, color = "grey", linestyle = ":", linewidth = 2)
plt.title("Pre-Training Eigenvalues")
plt.subplot(2,2,3)
plt.imshow(ij, cmap = "pink")
plt.colorbar()
plt.title("Post-Training IJ")
plt.subplot(2,2,4)
plt.scatter(x = np.linalg.eigvals(ij).real, y = np.linalg.eigvals(ij).imag, c = "Salmon", s = 20)
plt.axvline(x = 1, color = "grey", linestyle = ":", linewidth = 2)
plt.title("Post-Training Eigenvalues")

# ------------------------------------------------------------------------------

# Run trained network over time to compare to targets
PostRates = np.zeros([N,len(time)])
current = targets[:,0]
for t in np.arange(0,len(time)):
    PostRates[:,t] = np.tanh(current) # Add rate to traces
    weighted = np.matmul(ij,PostRates[:,t])
    current = (-current + weighted)*dt/tau + current # Update current
# Plots network rates after training
plt.figure(figsize = (23,3))
plt.imshow(np.arctanh(PostRates), cmap = "pink")
plt.ylabel("Currents")
plt.xlabel("Time (ms)")
plt.colorbar()
plt.show()
# Superimposed trained network neuron rates over target neuron rates
plt.subplots(nrows = 5, ncols = 1, figsize = (20,5))
for i in np.arange(0,5):
    plt.subplot(5,1,i+1)
    plt.plot(PostRates[i,:], linewidth = 1, color = "salmon")
    plt.plot(np.tanh(targets[i,:]), linewidth = 1, color = "darkblue")
    plt.ylabel("Rate")
    if i != 4:
        plt.tick_params(axis='x', bottom=False, labelbottom=False)
    if i == 0:
        plt.title("Red is Trained Network; Blue is Target")
plt.xlabel("Time (ms)")
