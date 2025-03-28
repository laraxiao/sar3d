import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt

def imgplot(img, title=''):
    # silhouette = True if img.shape[-1]==1 else False
    # inds = 0 if silhouette else range(3)
    plt.figure(figsize=(5, 5))
    # plt.imshow(img.cpu().detach().numpy())#,vmin=0, vmax=9, cmap='gray')
    plt.imshow(img.cpu().detach().numpy(), cmap='gray')
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.grid("off")
    plt.axis("off")
    plt.show()