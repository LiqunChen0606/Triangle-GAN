import random
import numpy as np
from math import *

import matplotlib
import pylab
import numpy as np
from PIL import Image
import matplotlib.patches as mpatches

def swiss_roll(batchsize, n_dim, n_labels):
	def sample(label, n_labels):
		uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
		r = sqrt(uni) * 3.0
		rad = np.pi * 4.0 * sqrt(uni)
		x = r * cos(rad)
		y = r * sin(rad)
		return np.array([x, y]).reshape((2,))

	z = np.zeros((batchsize, n_dim), dtype=np.float32)
	for batch in xrange(batchsize):
		for zi in xrange(n_dim / 2):
			z[batch, zi*2:zi*2+2] = sample(random.randint(0, n_labels - 1), n_labels)
	return z

def swiss_roll_test(batchsize, n_dim, n_labels):
	seq = np.linspace(0.0,1.0,num=batchsize)
	def sample(label, n_labels, batch):
		uni = seq[batch] #/ float(n_labels) + float(label) / float(n_labels)
		r = sqrt(uni) * 3.0
		rad = np.pi * 4.0 * sqrt(uni)
		x = r * cos(rad)
		y = r * sin(rad)
		return np.array([x, y]).reshape((2,))

	z = np.zeros((batchsize, n_dim), dtype=np.float32)
	for batch in xrange(batchsize):
		for zi in xrange(n_dim / 2):
			z[batch, zi*2:zi*2+2] = sample(random.randint(0, n_labels - 1), n_labels, batch)
	return z

def tile_images(image_batch, image_width=28, image_height=28, image_channel=1, dir=None, filename="images"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	pylab.gray()
	for m in range(100):
		pylab.subplot(10, 10, m + 1)
		pylab.imshow(image_batch[m].reshape((image_width, image_height)), interpolation="none")
		pylab.axis("off")
	pylab.savefig("{}/{}.png".format(dir, filename))

def plot_connect(orig,z_batch, a,b,c,d, dir=None, filename="z"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(5.0, 5.0)
	colors = ["#2103c8", "#0e960e", "#e40402","#05aaa8","#ac02ab","#aba808","#151515","#94a169", "#bec9cd", "#6a6551"]
	pylab.clf()
	for n in xrange(z_batch.shape[0]):
		#print np.floor_divide(n,(z_batch.shape[0]/10))
		result = pylab.scatter(orig[n, 0], 1, c=colors[np.floor_divide(n,1+(np.floor_divide(z_batch.shape[0],10)))], s=40, marker="o", edgecolors='none')
		result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[np.floor_divide(n,1+(np.floor_divide(z_batch.shape[0],10)))], s=40, marker="o", edgecolors='none')
	pylab.xlabel("z1")
	pylab.ylabel("z2")
	pylab.axis([a,b,c,d])
	pylab.savefig("{}/{}.png".format(dir, filename))


def plot_z(z_batch, a,b,c,d, dir=None, filename="z"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(5.0, 5.0)
	colors = ["#2103c8", "#0e960e", "#e40402","#05aaa8","#ac02ab","#aba808","#151515","#94a169", "#bec9cd", "#6a6551"]
	pylab.clf()
	for n in xrange(z_batch.shape[0]):
		#print np.floor_divide(n,(z_batch.shape[0]/10))
		#result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[np.floor_divide(n,1+(np.floor_divide(z_batch.shape[0],10)))], s=40, marker="o", edgecolors='none')
		result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[0], s=40, marker="o", edgecolors='none')
	pylab.xlabel("")
	pylab.ylabel("")
	pylab.axis([a,b,c,d])
	pylab.savefig("{}/{}.png".format(dir, filename))

def plot_labeled_z(z_batch, label_batch, dir=None, filename="labeled_z"):
	fig = pylab.gcf()
	fig.set_size_inches(20.0, 16.0)
	pylab.clf()
	colors = ["#2103c8", "#0e960e", "#e40402","#05aaa8","#ac02ab","#aba808","#151515","#94a169", "#bec9cd", "#6a6551"]
	for n in xrange(z_batch.shape[0]):
		result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[label_batch[n]], s=40, marker="o", edgecolors='none')

	classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
	recs = []
	for i in range(0, len(colors)):
		recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i]))

	ax = pylab.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(recs, classes, loc="center left", bbox_to_anchor=(1.1, 0.5))
	pylab.xticks(pylab.arange(-4, 5))
	pylab.yticks(pylab.arange(-4, 5))
	pylab.xlabel("z1")
	pylab.ylabel("z2")
	pylab.savefig("{}/{}.png".format(dir, filename))
