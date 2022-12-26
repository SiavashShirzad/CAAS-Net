from metrics import dice_coef


def dice_coef_loss(y_true, y_pred, smooth=1e-7):
    return 1 - dice_coef(y_true, y_pred, smooth)
