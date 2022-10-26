# -*- coding: utf-8 -*-
"""visualize the result."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.helpers import predict_knn


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
