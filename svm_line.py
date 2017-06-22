# Standard scientific Python imports
from PIL import Image
import os
from sklearn import svm, metrics
import numpy as np


def doSVM(img_train, target_train, img_test, target_test):

    classifier = svm.SVC(gamma=0.001)
    classifier.fit(img_train, target_train)
    expected = target_test
    predicted = classifier.predict(img_test)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s"
          % metrics.confusion_matrix(expected, predicted))

def prepareTraningDataSet(images_true, targets_true, images_false, targets_false , ratio) :
    img_train, target_train, img_test, target_test = [],[],[],[]

    for i in range(len(images_true)):
        if i < len(images_true)*ratio :
            img_train.append(images_true[i])
            target_train.append(1)
            img_train.append(images_false[i])
            target_train.append(0)
        else :
            img_test.append(images_true[i])
            target_test.append(1)
            img_test.append(images_false[i])
            target_test.append(0)

    return np.array(img_train) , np.array(target_train), np.array(img_test), np.array(target_test)

def main():
    # img_true_dir = "dataset/true_resize/"
    # img_false_dir = "dataset/false_resize/"
    img_true_dir = "dataset/true_resize_square/"
    img_false_dir = "dataset/false_resize_square/"
    images_true = []
    targets_true = []
    images_false = []
    targets_false = []

    # The dataset
    for image in os.listdir(img_true_dir):
        images_true.append(np.array(list(Image.open(img_true_dir + image).convert('L').getdata())))
        targets_true.append(1)

    i = 1
    n_samples = len(images_true)
    for image in os.listdir(img_false_dir):
        images_false.append(np.array(list(Image.open(img_false_dir + image).convert('L').getdata())))
        targets_false.append(0)
        i += 1
        if i > n_samples:
            break


    test_ratio = 0.9 # train_data : dataset = 9 : 10
    img_train , target_train, img_test, target_test= prepareTraningDataSet(images_true, targets_true, images_false, targets_false , test_ratio)

    # img_train = images[0::2]
    # target_train = targets[0::2]
    # img_test = images[1::2]
    # target_test = targets[1::2]

    print("len of img_train, target_train, img_test, target_test : %d, %d, %d, %d" % (len(img_train), len(target_train), len(img_test), len(target_test) ))
    doSVM(img_train, target_train, img_test, target_test)

if __name__ == '__main__' :
    main()