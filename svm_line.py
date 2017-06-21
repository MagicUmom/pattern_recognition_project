# Standard scientific Python imports
import matplotlib.pyplot as plt
from PIL import Image
import os

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
from collections import defaultdict
import numpy as np

img_true_dir = "dataset/true/"
img_false_dir = "dataset/false/"
images = np.array
targets = np.array
# The dataset
for image in os.listdir(img_true_dir):
    images = np.concatenate(images,np.array(list(Image.open(img_true_dir+image).convert('L').getdata())))
    targets = np.concatenate(targets,[1])
    # images_true.image.append(ns.array(list(Image.open(img_true_dir+image).convert('L').getdata())))
    # images_true.append(ns.array(list(Image.open(img_true_dir+image).convert('L').getdata())))

pixels = np.array
for pix in images :
    pixels.append(pix)

i = 1
n_samples = len(images)
for image in os.listdir(img_false_dir):
    images.append(np.array(list(Image.open(img_false_dir+image).convert('L').getdata())))
    target.append(0)
    i+=1
    if i > n_samples:
        break
data = images.reshape(n_samples,-1)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(dataset['image'][0::2], dataset['target'][0::2])

# Now predict the value of the digit on the second half:
expected = dataset['target'][1::2]
predicted = classifier.predict(dataset['image'][1::2])

print("Classification report for classifier %s:\n%s\n"
    % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s"
    % metrics.confusion_matrix(expected, predicted))
#
# images_and_predictions = list(
#                         zip(digits.images[int(n_samples / 2):], predicted))
# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Prediction: %i' % prediction)
#
# plt.show()
