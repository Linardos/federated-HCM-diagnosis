import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#sns.set_theme()
sns.set_style("white")
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

def cm_heatmap(confussion_matrix, classes, marker='s', title=None, grid_cols=10, save_path=None, \
                normalize=True, size_scale=None, hide_ticks=False):
    if normalize:
        norm_conf_matrix = confussion_matrix / confussion_matrix.astype(np.float).sum(axis=1)
        corr = pd.DataFrame(data=norm_conf_matrix, index=classes, columns=classes)
    else:
        corr = pd.DataFrame(data=confussion_matrix, index=classes, columns=classes)
    corr = pd.melt(corr.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    x=corr['y']
    y=corr['x']
    size=corr['value'].abs()
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    if size_scale is None:
        size_scale = int(1000/len(classes))
    else:
        size_scale = size_scale
        
    n_colors = 256 # Use 256 colors for the diverging color palette
    palette = sns.color_palette("Spectral", n_colors=256)
    color_min, color_max = [0, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1)) # target index in the color palette
        return palette[ind]

    # figsize=(800/my_dpi, 800/my_dpi),
    # plt.rcParams.update({'font.size': 6})
    # fig = plt.figure(constrained_layout=True)
    # gs0 = gridspec.GridSpec(1, 2, figure=fig)
    # gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[0])
    # for n in range(3):
    #     ax = fig.add_subplot(gs1[n])
    fig = plt.figure()
    if hide_ticks:
        fig.set_size_inches(10.5, 10.5, forward=True)
    plot_grid = plt.GridSpec(1, grid_cols, hspace=0.1, wspace=0.1, figure=fig) # Setup a 1xgrid_cols grid
    ax = plt.subplot(plot_grid[:,:-1], aspect='equal') # Use the leftmost (grid_cols-1) columns of the grid for the main plot
    color=corr['value']
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        c=size.apply(value_to_color),
        marker=marker # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=0, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    if hide_ticks:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.gca().invert_xaxis()
    if title is not None:
        ax.set_title(title)

    # Add color legend on the right side of the plot
    ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot
    col_x = [0]*len(palette) # Fixed x coordinate for the bars
    bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars
    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5]*len(palette), # Make bars 5 units wide
        left=col_x, # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False) # Hide grid
    ax.set_facecolor('white') # Make background white
    ax.set_xticks([]) # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right() # Show vertical ticks on the right

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()