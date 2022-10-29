# -*- coding: utf-8 -*-
"""visualize the result."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from src.utils import predict_knn
from src.test import get_predictions


def visualize_knn(y, x, k):
    step_size_x = 0.02
    step_size_y = 0.01
    min_max_x = 0.3
    min_max_y = 0.1

    background_color = ListedColormap(["#FFAAAA", "#AAAAFF"])
    points_color = ListedColormap(["#FF0000", "#0000FF"])

    print(x[:, 0].min())
    print(x[:, 0].max())
    # Start by getting teh decision boundaries
    min_x, max_x = x[:, 0].min(), x[:, 0].max()
    min_y, max_y = x[:, 1].min(), x[:, 1].max()
    x_mesh, y_mesh = np.meshgrid(
        np.arange(-min_max_x, min_max_x, step_size_x),
        np.arange(-min_max_y, min_max_y, step_size_y),
    )

    num_pts = x_mesh.shape[0] * x_mesh.shape[1]
    # Predict on the same data
    pred = predict_knn(y[:num_pts], x[:num_pts], x[:num_pts], k)

    # Plot the regions for each class
    pred = pred.reshape(x_mesh.shape)
    plt.figure()
    plt.pcolormesh(x_mesh, y_mesh, pred, cmap=background_color)

    pts = 100
    # Add in the data points
    plt.scatter(x[:pts, 0], x[:pts, 1], c=y[:pts], cmap=points_color)
    plt.xlim(x_mesh.min(), x_mesh.max())
    plt.ylim(y_mesh.min(), y_mesh.max())
    plt.title("KNN Class classification k = %i" % (k))

    plt.show()


def plot_features_covariance(x, headers):
    """
    Given the feature vector, plots a covariance heatmap amongst the features

    Args:
        x: shape(N, D) where N is the number of data samples and D is the dimension of each sample
        headers: list of strings of length D : Feature names

    Returns:
        A heatmap plot showing the covariance of a feature against each other feature
    """

    correlation_matrix = np.corrcoef(x.T)
    fig, axis = plt.subplots()
    plt.title("Correlation amongst input data features")

    image, colorbar = covariance_heatmap(
        correlation_matrix,
        headers,
        headers,
        axis,
        cmap="hot",
        vmin=-1,
        vmax=1,
        colorbar_label="Correlation coefficient",
    )
    texts = add_text_heatmap(image, size=7)

    fig.tight_layout()
    plt.show()


def covariance_heatmap(
    data,
    row_labels,
    col_labels,
    axis=None,
    colorbar_properties=None,
    colorbar_label="",
    **kwargs
):

    image = axis.imshow(data, **kwargs)
    if colorbar_properties == None:
        colorbar_properties = dict()

    # Create a vertical bar to show the relation between color shade and values
    colorbar = axis.figure.colorbar(image, ax=axis, **colorbar_properties)
    colorbar.ax.set_ylabel(colorbar_label, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    axis.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    axis.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Align the bottom labels for better readability
    plt.setp(axis.get_xticklabels(), rotation=-30, ha="left", rotation_mode="anchor")

    axis.spines[:].set_visible(False)

    axis.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)

    axis.grid(which="minor", color="black", linestyle="-", linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)

    return image, colorbar


def add_text_heatmap(image, data=None, dataformat="{x:.2f}", **textkw):

    data = image.get_array()

    # Set aligment to be center
    cell_properties = {"horizontalalignment": "center", "verticalalignment": "center"}

    cell_properties.update(textkw)

    # For keeping cell values in a specific format
    formatter = matplotlib.ticker.StrMethodFormatter(dataformat)

    # Apply formatting to each cell
    cell_values = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            cell_properties.update(color="black")
            cell_value = image.axes.text(
                j, i, formatter(data[i, j], None), **cell_properties
            )
            cell_values.append(cell_value)

    return cell_values


def plot_train_validation_losses(
    loss_tr, loss_val, loss_function_name="Cross entropy loss"
):
    """
    Plots the training and validation losses over multiple training iterations to check
    if the model overfits on training samples
    """
    # Number of training iterations
    num_iterations = np.arange(len(loss_tr))

    # Plot the loss values in the same figure
    plt.plot(num_iterations, loss_tr, marker=".", color="b", label="train error")
    plt.plot(num_iterations, loss_val, marker=".", color="r", label="validation error")

    # Re-scale
    plt.xticks(np.arange(min(num_iterations) - 1, max(num_iterations) + 1, 50))
    plt.xlabel("train iteration")
    plt.ylabel(loss_function_name)
    plt.title("Loss decomposition over training iterations")

    plt.legend(loc=2)
    plt.grid(True)

    # Show visualization
    plt.show()
