#How to load Images and display them using 
#Python, tensorflow

#!/home/sishel/tensorflow/venv/bin/python


from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt
import os



DB_DIR = '/home/sishel/FingerprintDatabase/'
DIR = '/home/sishel/Code/DeepClassifier/'
IMG_H = 64
IMG_W = 64
IMG_CH = 1


def makeImgList():
	if (os.path.isdir(DB_DIR) == False):
		print('{} does not exist'.format(DB_DIR))
	else:
		wholeImgList = []
		for mainDirName, subDirs, files in os.walk(DB_DIR):
			print('mainDirName: {}'.format(mainDirName))
			print('subDirs: {}'.format(subDirs))
			i = 1
			for name in files:
				imgFileName = DB_DIR + name
				wholeImgList.append(imgFileName)
				print('{}. {}'.format(i, imgFileName))
				i += 1
	return wholeImgList


def loadImage(imgList):
	lArr = len (imgList)
	imgArray = np.ones ((lArr, IMG_H, IMG_W, IMG_CH)) 
	for i in range(len(imgList)):
		img = load_img(
					imgList[i],
					grayscale = True,
					target_size = [IMG_H, IMG_W],
					interpolation='bicubic'
				)
		imgArray[i] = np.reshape(img, [IMG_H, IMG_W, IMG_CH])
	return imgArray

	
imgList = makeImgList()
imgArray = loadImage(imgList)


def displayImage(imgSet, title):
	fig = plt.figure(figsize = (20,20))
	plt.suptitle(title)
	for i in range(16):
		subPlt = fig.add_subplot(4, 4, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(imgSet[i, :, :, 0], cmap = plt.get_cmap('gray'))

	plt.show()


displayImage(imgArray, "Test")
