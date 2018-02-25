import matplotlib.pyplot as plt

def plot_image(img, cmap=None):
    # if you wanted to show a single color channel image called 'gray'
    # for example, call as plt.imshow(gray, cmap='gray')
    plt.imshow(img, cmap)
    plt.show()


def plot_2_images(img, img2):
    # if you wanted to show a single color channel image called 'gray'
    # for example, call as plt.imshow(gray, cmap='gray')
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Image 1', fontsize=50)
    ax2.imshow(img2)
    ax2.set_title('Image 2', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
