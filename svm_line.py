# Standard scientific Python imports
from PIL import Image
import os
from sklearn import svm, metrics
import numpy as np


def doSVM(img_train, target_train, img_test, target_test):

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    # We learn the digits on the first half of the digits
    classifier.fit(img_train[0::2], target_train[0::2])

    # Now predict the value of the digit on the second half:
    expected = target_test[1::2]
    predicted = classifier.predict(img_test[1::2])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s"
          % metrics.confusion_matrix(expected, predicted))



def main():
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
        images.append(np.array(list(Image.open(img_false_dir + image).convert('L').getdata())))
        targets.append(0)
        i += 1
        if i > n_samples:
            break
    images = np.array(images)
    targets = np.array(targets)

    img_train = images[0::2]
    target_train = targets[0::2]
    img_test = images[1::2]
    target_test = targets[1::2]

    print("len of img_train, target_train, img_test, target_test : %d, %d, %d, %d" % (len(img_train), len(target_train), len(img_test), len(target_test) ))
    doSVM(img_train, target_train, img_test, target_test)

if __name__ == '__main__' :
    main()