# Thankfully to ultralytics' convenient wrapper, we can simply use a dict

AUGMENTATIONS = {
    "hsv_h": 0.015,      # Hue
    "hsv_s": 0.7,        # Saturation
    "hsv_v": 0.4,        # Value
    "degrees": 25.0,     # Random rotation
    "translate": 0.1,    # Translate by +-10% of image size
    "scale": 0.5,        # Resize scaling by 50% (both directions)
    "shear": 2.0,        # Shear by ±2 degrees
    "flipud": 0.5,       # Vertical flip
    "fliplr": 0.5,       # Horizontal flip prob
    "mosaic": 1.0,       # Mosaic enabled (good for small object generalization)
    "mixup": 0.1,        # Light mixup to regularize
}
