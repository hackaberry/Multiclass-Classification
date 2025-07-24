# Author: Anthony Madrigal Calderon
# Date: 07/23/2025

import torch
from torch import nn
import tkinter as tk
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from ClassificationModel import ClassificationModel
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)

# Update accuracy widget
def update_acc(accuracy):
	acc_label.config(text=f"Accuracy: {accuracy:.2f}%")

# Loss curve graph
def plot_loss(x,y):
	loss_graph.clear()
	loss_graph.scatter(x,y,c="0")
	canvas.draw()

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
	# Credit to Daniel Bourke: https://github.com/mrdbourke. Minor modifications were made

    # Put everything to CPU 
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    plot_decision_boundary_graph.clear()    
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plot_decision_boundary_graph.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plot_decision_boundary_graph.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    canvas.draw()

# This function returns the accuracy of the model
def accuracy_fn(preds,labels):
	correct = torch.eq(preds,labels).sum().item()
	acc = (correct/len(preds)) * 100
	return acc

# This is the training/testing loop that is called through the button presses. It invokes other functions
def training_loop(model,next_n_epochs):
	global epoch # For tracking the x-values of the loss curve

	for x in range(next_n_epochs):

		model.train()

		train_logits = model(X_train)
		train_labels = torch.softmax(train_logits,dim=1).argmax(dim=1) # logits -> probabilities -> labels

		loss = loss_fn(train_logits,y_train)
		train_acc = accuracy_fn(train_labels,y_train) # Tracking model accuracy

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# For drawing loss curve
		epoch += 1
		epoch_count.append(epoch)
		loss_count.append(loss.item())

	# Testing 
	model.eval()
	with torch.inference_mode():
		test_logits = model(X_test)
		test_labels = torch.softmax(test_logits,dim=1).argmax(dim=1)
		test_acc = accuracy_fn(test_labels,y_test)

	# Drawing decision boundary, loss curve and updating accuracy label
	plot_decision_boundary(model, X_test, y_test)
	plot_loss(epoch_count,loss_count)
	update_acc(test_acc)


# Constants
N_SAMPLES = 1000
N_FEATURES = 2
N_CLASSES = 4
RANDOM_SEED = 1
HIDDEN_UNITS = 5
IN_FEATURES = 2
OUT_FEATURES = 4

# For loss graph
epoch = 0
epoch_count = []
loss_count = []

# Device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ClassificationModel(IN_FEATURES,OUT_FEATURES,HIDDEN_UNITS).to(device=device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

# Dataset consists of sklearn "make_blobs" toy dataset
X, y = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=RANDOM_SEED)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=RANDOM_SEED)

X_train,y_train = torch.from_numpy(X_train).type(torch.float), torch.from_numpy(y_train).type(torch.long)
X_test,y_test = torch.from_numpy(X_test).type(torch.float), torch.from_numpy(y_test).type(torch.long)

X_train, y_train = X_train.to(device=device), y_train.to(device=device)
X_test, y_test = X_test.to(device=device), y_test.to(device=device)

# GUI initialization
root = tk.Tk()
root.title("Learn2Classify")
root.geometry("1000x1000")
root.configure(bg="lightgray")

# Graph initialization
fig = plt.figure(figsize=(10,7))
plot_decision_boundary_graph = fig.add_subplot(1, 2, 1) # Left graph
loss_graph = fig.add_subplot(1,2,2) # Right graph
canvas = FigureCanvasTkAgg(fig, master=root) # Embed graphs in Tk window
plot_decision_boundary_graph.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap="viridis")
plot_decision_boundary_graph.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap="viridis")
canvas.get_tk_widget().pack()
canvas.draw()

# Widgets
next_10_epochs = tk.Button(root,text="Next 10 epochs",bg="lightblue",command=lambda: training_loop(model,10))
next_50_epochs = tk.Button(root,text="Next 50 epochs",bg="lightblue",command=lambda: training_loop(model,50))

next_10_epochs.place(relx=0.43,rely=0.065)
next_50_epochs.place(relx=0.53,rely=0.065)

acc_label = tk.Label(root,text="Accuracy:",  font=("Arial", 14))
acc_label.place(relx=0.47,rely=0.020)

root.mainloop()