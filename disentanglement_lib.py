import imageio
import torch
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import scipy


def cycle_interval(starting_value, num_frames, min_val, max_val):
    """Cycles through the state space in a single cycle."""
    starting_in_01 = ((starting_value - min_val) / (max_val - min_val)).cpu()
    grid = torch.linspace(starting_in_01.item(), starting_in_01.item() + 2., steps=num_frames + 1)[:-1]
    grid -= np.maximum(0, 2 * grid - 2)
    grid += np.maximum(0, -2 * grid)
    return grid * (max_val - min_val) + min_val


def save_animation(list_of_animated_images, image_path, fps):
    full_size_images = []
    for single_images in zip(*list_of_animated_images):
        full_size_images.append(list(single_images))
    imageio.mimwrite(image_path, full_size_images, fps=fps)


def padding_array(image, padding_px, axis, value=None):
    """Creates padding image of proper shape to pad image along the axis."""
    shape = list(image.shape)
    shape[axis] = padding_px
    if value is None:
        return np.ones(shape, dtype=image.dtype)
    else:
        assert len(value) == shape[-1]
        shape[-1] = 1
        return np.tile(value, shape)


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                 dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = GradientBoostingClassifier()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def rescale_dsprites_latents(latent_train, latent_val):
    """ Don't look """
    """ Necessary for classifier in dci"""
    latent_train[:, 2] = ((latent_train[:, 2] - 0.5) * 10).int()
    latent_val[:, 2] = ((latent_val[:, 2] - 0.5) * 10).int()
    latent_train[:, 3] = (latent_train[:, 3] * 20 / np.pi).int()
    latent_val[:, 3] = (latent_val[:, 3] * 20 / np.pi).int()
    latent_train[:, 4] = (32 * latent_train[:, 4]).int()
    latent_val[:, 4] = (32 * latent_val[:, 4]).int()
    latent_train[:, 5] = (32 * latent_train[:, 5]).int()
    latent_val[:, 5] = (32 * latent_val[:, 5]).int()
    return latent_train, latent_val


def compute_importance_gbt(mu_train, latent_train, mu_val, latent_val):
    mu_train, latent_train = mu_train.numpy(), latent_train.numpy()
    mu_val, latent_val = mu_val.numpy(), latent_val.numpy()
    num_factors = latent_train.shape[1]
    num_mu = mu_train.shape[1]
    importance_matrix = np.zeros(shape=[num_mu, num_factors],
                                 dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(1, num_factors):
        model = GradientBoostingClassifier()
        model.fit(mu_train, latent_train[:, i])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(mu_train) == latent_train[:, i]))
        test_loss.append(np.mean(model.predict(mu_val) == latent_val[:, i]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def compute_dci(mus_train, latents_train, mus_test, latents_test, load=False):
    """Computes score based on both training and testing codes and factors."""
    scores = {}

    if load:
        importance_matrix = np.load('./checkpoints/importance_matrix.npy')
        train_err = 0.624 #np.load('./checkpoints/train_err.npy')
        test_err = 0.595 #np.load('./checkpoints/test_err.npy')
    else:  # Since this shit takes a long time
        importance_matrix, train_err, test_err = compute_importance_gbt(
            mus_train, latents_train, mus_test, latents_test)
        np.save('./checkpoints/importance_matrix.txt', importance_matrix)
        np.save('./checkpoints/train_err.txt', train_err)
        np.save('./checkpoints/test_err.txt', test_err)

    assert importance_matrix.shape[0] == mus_train.shape[1]
    assert importance_matrix.shape[1] == latents_train.shape[1]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores, importance_matrix


def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_mu = disentanglement_per_mu(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_mu * code_importance)


def disentanglement_per_mu(importance_matrix):
    """Compute disentanglement score of each mu."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])
