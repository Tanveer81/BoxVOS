import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def visualize_saliency(model):
	# Initialize the model
	# model = torch.load('/home/koner/barlow_twins_video/checkpoint/weights/checkpoint.pkl')

	# Set the model to run on the GPU
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	# Set the model on Eval Mode
	model.eval()



	# Open the image file
	image = Image.open('/nfs/data3/koner/data/youtubeVOS/train/JPEGImages/04990d1915/00015.jpg')

	# Set up the transformations
	transform_ = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
	])

	# Transforms the image
	image = transform_(image)


	# Reshape the image (because the model use
	# 4-dimensional tensor (batch_size, channel, width, height))
	image = image.reshape(1, 3, 224, 224)

	# Set the device for the image
	image = image.to(device)

	# Set the requires_grad_ to the image for retrieving gradients
	image.requires_grad_()

	# Retrieve output from the image
	output = model(image)['p7']

	# Catch the output
	output_idx = output.argmax()
	output_max = output[0, output_idx]

	# Do backpropagation to get the derivative of the output based on the image
	output_max.backward()

	# Retireve the saliency map and also pick the maximum value from channels on each pixel.
	# In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
	saliency, _ = torch.max(image.grad.data.abs(), dim=1)
	saliency = saliency.reshape(224, 224)

	# Reshape the image
	image = image.reshape(-1, 224, 224)

	# Visualize the image and the saliency map
	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
	ax[0].axis('off')
	ax[1].imshow(saliency.cpu(), cmap='hot')
	ax[1].axis('off')
	plt.tight_layout()
	fig.suptitle('The Image and Its Saliency Map')
	plt.show()