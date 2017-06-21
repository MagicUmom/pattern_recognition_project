# Standard scientific Python imports
import matplotlib.pyplot as plt
from PIL import Image
import os

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
import numpy as np

img_true_dir = "dataset/true/"
img_false_dir = "dataset/false/"
images_true =[]
images_false =[]

# The dataset

for image in os.listdir(img_true_dir):
    images_true.append(Image.open(img_true_dir+image))

for image in os.listdir(img_false_dir):
    images_false.append(Image.open(img_false_dir+image))



# images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:4]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
np.array(list(images_true.convert('L').getdata()))

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:int(n_samples / 2)], digits.target[:int(n_samples / 2)])

# Now predict the value of the digit on the second half:
expected = digits.target[ int(n_samples / 2):]
predicted = classifier.predict(data[ int(n_samples / 2):])

print("Classification report for classifier %s:\n%s\n"
    % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s"
    % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(
                        zip(digits.images[int(n_samples / 2):], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
