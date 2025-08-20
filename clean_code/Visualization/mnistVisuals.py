import matplotlib.pyplot as plt
import numpy as np
from typing import List

from Visualization import visualHelpers


def _prepare_image_rgba(digit_array, color, alpha_actual):
    alpha_mask = np.clip(digit_array.reshape(28, 28), 0, 255) / 255
    rgb_image = np.zeros((28, 28, 4))
    for c in range(3):
        rgb_image[..., c] = color[c]
    rgb_image[..., 3] = alpha_mask * alpha_actual
    return rgb_image


def _plot_digits(transformed_points, point_colors, data, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter):
    landmark_indicies = [landmark.index for landmark in voltages.get_all_landmarks()]
    drawn_xy = []
    size_sqrd = image_size ** 2
    for i in range(transformed_points.shape[0]):
        color = point_colors[i]
        size = landmarkSize if i in landmark_indicies else 1
        rgba_img = _prepare_image_rgba(data[i], color, alpha_actual)
        x, y = transformed_points[i]

        draw = True
        if remove_clutter:
            for (x2, y2) in drawn_xy:
                if ((x2 - x) ** 2 + (y2 - y) ** 2 < size_sqrd):
                    draw = False
                    break

        if draw:
            if remove_clutter:
                drawn_xy.append((x, y))

            ax.imshow(
                rgba_img,
                extent=(x - image_size * size, x + image_size * size, y - image_size * size, y + image_size * size),
                origin='upper'
            )


def plot_mnist_unlabeled(voltages, data, transformation="mds", landmarkSize=3, alpha_actual=1,
                         percent_size=0.02, argmax=True, remove_clutter=True, out_file=None):
    """
    Visualizes MNIST digits in 2D space after dimensionality reduction (MDS or PCA),
    coloring and sizing them based on their voltage values.
    """
    points = voltages.voltage_array()
    transformed_points = visualHelpers.transform(points, transformation)
    x_bound, y_bound, image_size = visualHelpers.compute_image_size(transformed_points, percent_size)
    fig, ax = visualHelpers.setup_figure(x_bound, y_bound, image_size, landmarkSize, "Visualization of K-Means MNIST")

    colors = visualHelpers.get_distinct_colors(points[0].shape[0])
    point_colors = [colors[np.argmax(p)] for p in points] if argmax else [colors[np.argmin(p)] for p in points]

    _plot_digits(transformed_points, point_colors, data, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter)
    visualHelpers.standard_save_display(out_file)


def plot_mnist_digits(voltages, data, labels, transformation="mds", landmarkSize=3, alpha_actual=1,
                      percent_size=0.02, remove_clutter=True, out_file=None):
    """
    Visualizes MNIST digits in 2D space using voltage-based embeddings reduced by PCA or MDS.
    Each digit is rendered as a translucent RGB image, colored by its true label.
    """
    points = voltages.voltage_array()
    transformed_points = visualHelpers.transform(points, transformation)
    x_bound, y_bound, image_size = visualHelpers.compute_image_size(transformed_points, percent_size)
    fig, ax = visualHelpers.setup_figure(x_bound, y_bound, image_size, landmarkSize, "Visualization of Digits")

    colors = visualHelpers.get_distinct_colors(len(set(labels)))
    point_colors = [colors[int(l)] if l is not None else (1, 1, 1) for l in labels]

    _plot_digits(transformed_points, point_colors, data, voltages, ax, image_size, landmarkSize, alpha_actual, remove_clutter)
    visualHelpers.standard_save_display(out_file)


def compute_labels(label_counts, ratio_threshold=0.6, size_threshold=5):
    """
    Compute labels based on the label counts, ratio threshold, and size threshold.
    """
    str_labels = []
    for label_count in label_counts:
        if label_count is None:
            label = "small"
        else:
            common = label_count.most_common()
            total_count = sum([c[1] for c in common])
            ratio = common[0][1] / total_count
            if total_count < size_threshold:
                label = "small"
            else:
                if ratio > ratio_threshold:
                    label = common[0][0]
                else:
                    label = "weak_maj"
        str_labels.append(label)
    return str_labels


def scatter_plot(points, transformed_points, data, focus_on, labels, reverse_dict_labels,
                 percent_size=0.01, alpha_actual=1, out_file=None, element="digit"):
    """
    Creates a scatter plot of transformed points with digit images or points.
    """
    if out_file is None:
        out_file = "mnist_visualization.png"

    fig, ax = plt.subplots(figsize=(12, 10))

    from Visualization.visualHelpers import generate_vivid_colors
    colors = generate_vivid_colors(len(reverse_dict_labels))

    x_bound = (transformed_points[:, 0].min(), transformed_points[:, 0].max())
    y_bound = (transformed_points[:, 1].min(), transformed_points[:, 1].max())
    image_size = (x_bound[1] + y_bound[1] - x_bound[0] - y_bound[0]) * percent_size / 2

    count_nones = 0

    for i in range(transformed_points.shape[0]):
        point_voltages = points[i, :]
        label = labels[i]

        if (label is not None) and (label != 0):
            size = 1
            color = np.array(colors[int(label)])
            x, y = transformed_points[i]

            if (np.min(point_voltages) == 0.0):
                min_index = np.argmin(point_voltages)
                min_index = focus_on[min_index]
                plt.text(x, y, str(min_index), fontsize=20, color='white', ha='center', va='center')

            if label == 1:  # weak_maj
                plt.plot(x, y, marker='o', markersize=1, color=color)
            else:
                if element == "digit":
                    rgb_image = _prepare_image_rgba(data[i], color, alpha_actual)
                    ax.imshow(rgb_image, extent=(x - image_size * size, x + image_size * size,
                                                 y - image_size * size, y + image_size * size), origin='upper')
                elif element == "point":
                    plt.plot(x, y, marker='o', markersize=6, color=color)
                else:
                    raise ValueError("element must be either 'digit' or 'point'")
        else:
            count_nones += 1

    ax.set_xlim(x_bound[0] - image_size, x_bound[1] + image_size)
    ax.set_ylim(y_bound[0] - image_size, y_bound[1] + image_size)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    plt.title("Visualization of Digits")

    print(f"Number of points with no label: {count_nones}")
    visualHelpers.standard_save_display(out_file)


def plot_landmark_subset(points, centroids, label_counts, focus_on=None, log_transform=True,
                         transformation='pca', **kwargs):
    """
    Visualizes a subset of points in 2D space after dimensionality reduction, focusing on specific landmarks.
    """
    if focus_on is None:
        focus_on = np.array(range(points.shape[1]), **kwargs)
    if log_transform:
        points = -np.log(points)

    str_labels = compute_labels(label_counts)
    possible_labels = set(str_labels) - set(['small', 'weak_maj'])
    dict_labels = {label: i + 2 for i, label in enumerate(sorted(list(possible_labels)))}
    dict_labels['small'] = 0
    dict_labels['weak_maj'] = 1
    labels = [dict_labels[label] for label in str_labels]
    reverse_dict_labels = {value: key for key, value in dict_labels.items()}

    closest_landmarks = np.argmin(points, axis=1)
    mask = np.isin(closest_landmarks, focus_on)

    points = points[mask, :]
    points = points[:, focus_on]
    labels = np.array(labels)[mask]
    centroids = centroids[mask, :]

    transformed_points = visualHelpers.transform(points, transformation)
    scatter_plot(points, transformed_points, centroids, focus_on, labels, reverse_dict_labels, **kwargs)