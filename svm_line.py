# Standard scientific Python imports
import matplotlib.pyplot as plt
from PIL import Image
import os

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
from collections import defaultdict
import numpy as np

img_true_dir = "dataset/true_resize/"
img_false_dir = "dataset/false_resize/"
images = []
targets = []
# The dataset
for image in os.listdir(img_true_dir):
    images.append(np.array(list(Image.open(img_true_dir + image).convert('L').getdata())))
    targets.append(1)

i = 1
n_samples = len(images)
for image in os.listdir(img_false_dir):
    images.append(np.array(list(Image.open(img_false_dir+image).convert('L').getdata())))
    targets.append(0)
    i+=1
    if i > n_samples:
        break

images = np.array(images)
targets = np.array(targets)


data = images.reshape(n_samples,-1)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(images[0::2], targets[0::2])

# Now predict the value of the digit on the second half:
expected = targets[1::2]
predicted = classifier.predict(images[1::2])

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
