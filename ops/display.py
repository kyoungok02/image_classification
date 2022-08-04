import matplotlib.pyplot as plt

def show(imgs, y=None, color=True):
    for i, img in enumerate(imgs):
        npimg = img.numpy()
        npimg_tr = np.transpose(npimg, (1, 2, 0))
        plt.subplot(1, imgs.shape[0], i+1)
        plt.imshow(npimg_tr)
    
    # plt.imshow(npimg_tr)
    if y is not None:
        plt.title('labels: ' + str(y))