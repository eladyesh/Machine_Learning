# seaborn library
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.rand(10, 10)

# Create a figure with subplots for each colormap
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.flatten()

# Define the colormaps to use
cmaps = ['Blues', 'BuGn', 'BuPu', 'Greens', 'Greys', 'Oranges', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd']

# Plot the heatmap for each colormap
for i, cmap in enumerate(cmaps):
    sns.heatmap(data, ax=axs[i], cmap=cmap)
    axs[i].set_title(cmap)

    # Show the plot
    plt.show()
