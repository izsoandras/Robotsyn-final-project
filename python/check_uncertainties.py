import numpy as np
import cv2
import os.path
import matplotlib.pyplot as plt


def run():
    sample_num = 10
    folder = '../data_hw5_ext/calibration'
    im_path = '../data_hw5_ext/IMG_8207.jpg'

    img = cv2.imread(im_path)

    K           = np.loadtxt(os.path.join(folder, 'K.txt'))
    dc          = np.loadtxt(os.path.join(folder, 'dc.txt'))
    std_int     = np.loadtxt(os.path.join(folder, 'std_int.txt'))

    # extract std deviations
    fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy = std_int

    rands = np.random.normal(dc[[0,1,4,2,3]], [k1, k2, k3, p1, p2], size=(sample_num,1,5)).squeeze()

    undist_imgs = []

    for idx, sample in enumerate(rands):
        newCameraMatrix,_ = cv2.getOptimalNewCameraMatrix(K, sample[[0, 1, 3, 4, 2]], img.shape[0:2], 1)
        undist_img = cv2.undistort(img, K, sample[[0,1,3,4,2]],newCameraMatrix=newCameraMatrix)
        undist_imgs.append( undist_img )
        cv2.imwrite(f'./undist/image_{idx}.jpg', undist_img)

    plt.figure()
    for i in range(2):
        for j in range(5):
            plt.subplot(2,5, i*5+j+1)
            plt.imshow(undist_imgs[i*5+j])

    plt.show()

if __name__ == "__main__":
    run()