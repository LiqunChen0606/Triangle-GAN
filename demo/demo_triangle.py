from math import *
import pylab
import numpy as np
from numpy import random as nrandom
from PIL import Image
import matplotlib.patches as mpatches

import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable

from helpers import plot_z
from itertools import chain

mb_size = 512 	#batch size
x_dim = 1	#data dimension
X_dim = 3	#augmented data dimension
j_dim = 2	#joint data dimension
h_dim = 512	#hidden layer dimension
lr = 1*1e-5	#learning rate
noise = 1	#noise on samples
n_critic = 3	#inner loop nbr of dircriminator

data_nbr = 20000

dtype = torch.cuda.FloatTensor

G_AB = torch.nn.Sequential(
	torch.nn.Linear(X_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, x_dim),
)

G_BA = torch.nn.Sequential(
	torch.nn.Linear(X_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, x_dim),
)

D_A = torch.nn.Sequential(
	torch.nn.Linear(j_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, 1),
	torch.nn.Sigmoid(),
)

D_B = torch.nn.Sequential(
	torch.nn.Linear(j_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, h_dim),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(h_dim, 1),
	torch.nn.Sigmoid(),
)

G_AB.cuda()
G_BA.cuda()
D_A.cuda()
D_B.cuda()

def reset_grad():
	G_AB.zero_grad()
	G_BA.zero_grad()
	D_A.zero_grad()
	D_B.zero_grad()

def get_minibatches_idx(n, minibatch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


bce = torch.nn.BCELoss().cuda()

G_solver = optim.Adam(chain(G_AB.parameters(),G_BA.parameters()), lr=lr)
D_solver = optim.Adam(chain(D_A.parameters(),D_B.parameters()), lr=lr)

mean2 = [0,1.5]
mean4 = [-1.5,0]
mean6 = [1.5,0]
mean8 = [0,-1.5]

cov2 = [[3,0],[0,0.025]]
cov4 = [[0.025,0],[0,3]]
cov6 = [[0.025,0],[0,3]]
cov8 = [[3,0],[0,0.025]]
x2,y2 = np.random.multivariate_normal(mean2,cov2,5000).T
x4,y4 = np.random.multivariate_normal(mean4,cov4,5000).T
x6,y6 = np.random.multivariate_normal(mean6,cov6,5000).T
x8,y8 = np.random.multivariate_normal(mean8,cov8,5000).T


X2 = np.stack((x2,y2)).T
X4 = np.stack((x4,y4)).T
X6 = np.stack((x6,y6)).T
X8 = np.stack((x8,y8)).T


toy_data = np.concatenate((X2,X4,X6,X8))
np.random.shuffle(toy_data)

real_label = 1
fake_label = 0
real_tensor = Variable(torch.FloatTensor(mb_size)).cuda()
_ = real_tensor.data.fill_(real_label)

paired_real_tensor = Variable(torch.FloatTensor(mb_size)).cuda()
_ = paired_real_tensor.data.fill_(real_label)

fake_tensor = Variable(torch.FloatTensor(mb_size)).cuda()
_ = fake_tensor.data.fill_(fake_label)


for it in range(0,1000):
	kf = get_minibatches_idx(data_nbr, mb_size)
	for i in range(data_nbr/mb_size):

		critic = n_critic

		x_n1 = np.random.normal(0, noise, (mb_size,x_dim))
		x_n1 = Variable(torch.from_numpy(x_n1).cuda())
		x_n1 = x_n1.float()

		x_n2 = np.random.normal(0, noise, (mb_size,x_dim))
		x_n2 = Variable(torch.from_numpy(x_n2).cuda())
		x_n2 = x_n2.float()

		x_A = toy_data[kf[i][1]][:,0].reshape((-1,1))
		x_A = Variable(torch.from_numpy(x_A).cuda())
		x_A = x_A.float()
		x_An = torch.cat((x_n1,x_n2,x_A),1)

		x_B = toy_data[kf[i][1]][:,1].reshape((-1,1))
		x_B = Variable(torch.from_numpy(x_B).cuda())
		x_B = x_B.float()
		x_Bn = torch.cat((x_n1,x_n2,x_B),1)

		x_real   = torch.cat((x_A,x_B),1)

		for _ in range(critic):
		# Sample data
			reset_grad()
			# Dicriminator forward-loss-backward-update
			x_AB = G_AB(x_An).detach()
			x_BA = G_BA(x_Bn).detach()

			x_fakeAB = torch.cat((x_A,x_AB),1)
			x_fakeBA = torch.cat((x_BA,x_B),1)

		        l_d_A_real, l_d_A_fake1, l_d_A_fake2 = bce(D_A(x_real), paired_real_tensor), bce(D_A(x_fakeAB), fake_tensor), bce(D_A(x_fakeBA), fake_tensor)
		        l_d_B_real, l_d_B_fake = bce(D_B(x_fakeAB), real_tensor), bce(D_B(x_fakeBA), fake_tensor)


			l_d_A = l_d_A_real + l_d_A_fake1 + l_d_A_fake2
			l_d_B = l_d_B_real + l_d_B_fake

			l_d = l_d_A + l_d_B

			l_d.backward()

			D_solver.step()

			# Housekeeping - reset gradient
			reset_grad()

		# Generator forward-loss-backward-update
		x_AB = G_AB(x_An)
		x_BA = G_BA(x_Bn)
		x_fakeAB = torch.cat((x_A,x_AB),1)
		x_fakeBA = torch.cat((x_BA,x_B),1)

                l_gan_A = bce(D_A(x_fakeAB), real_tensor) + bce(D_B(x_fakeAB), fake_tensor)
                l_gan_B = bce(D_A(x_fakeBA), real_tensor) + bce(D_B(x_fakeBA), real_tensor)

		l_g = l_gan_A + l_gan_B
		l_g.backward()
		G_solver.step()

		# Housekeeping - reset gradient
		reset_grad()

	# Print and plot every now and then
	print('Iter-{}; D_loss: {}; Gen_loss: {}'
	      	.format(it, l_d.data[0], l_g.data[0]))
	sample_x_real = x_real.cpu().data.numpy()
	plot_z(sample_x_real,-3,3,-3,3,'./triangleGAN','real_samples')
	sample_x_AB = x_fakeAB.cpu().data.numpy()
	sample_x_BA = x_fakeBA.cpu().data.numpy()
	plot_z(sample_x_AB,-3,3,-3,3,'./triangleGAN','fake_AB_in_'+str(it))
	plot_z(sample_x_BA,-3,3,-3,3,'./triangleGAN','fake_BA_in_'+str(it))
