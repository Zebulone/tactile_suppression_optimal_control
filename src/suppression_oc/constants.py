import seaborn as sns
from suppression_oc.oc.models.todorov import DEFAULT_SIGMA_POS

# screen measurements in m
SCREEN_HEIGHT = 0.29

# pixels on the screen
SCREEN_RESOLUTION = (1920, 1080)

# frame rate of the motion capture system in Hz
FRAME_RATE = 100

# plotting
CONDITIONS = ["normal", "uncertain", "no_vision"]

CONDITION_LABELS = ["certain", "uncertain", "no-vision"]

CONDITION_LABELS_DICT = {
    "normal": "certain",
    "uncertain": "uncertain",
    "no_vision": "no-vision",
}

CONDITION_PALETTE = sns.color_palette("colorblind", n_colors=3)

CONDITION_COLORS = {
    "normal": CONDITION_PALETTE[0],
    "uncertain": CONDITION_PALETTE[1],
    "no_vision": CONDITION_PALETTE[2],
}

CONDITION_LINESTYLES = {
    "normal": "-",
    "uncertain": "--",
    "no_vision": ":",
}

BINS = [-float("inf"), -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1.0, float("inf")]

TIMEPOINT_ORDER = [
    "(-inf, -0.1]",
    "(-0.1, 0.0]",
    "(0.0, 0.2]",
    "(0.2, 0.4]",
    "(0.4, 0.6]",
    "(0.6, 0.8]",
    "(0.8, 1.0]",
    "(1.0, inf]",
]


TIMEPOINT_LABELS = [
    "[<-.1]s",
    "(-.1, .0]s",
    "(.0, .2]",
    "(.2, .4]",
    "(.4, .6]",
    "(.6, .8]",
    "(.8, 1.0]",
    "[>1.0]",
]

# movement-time fractions covered by each movement bin
MOVE_BINS = {
    "(0.0, 0.2]": (0.0, 0.2),
    "(0.2, 0.4]": (0.2, 0.4),
    "(0.4, 0.6]": (0.4, 0.6),
    "(0.6, 0.8]": (0.6, 0.8),
    "(0.8, 1.0]": (0.8, 1.0),
}


CM_TO_INCHES = 1 / 2.54

# optimal control model initial uncertainty parameters
INIT_SIGMA_POS_NORMAL = 0.0083
INIT_SIGMA_POS_UNCERTAIN = 0.0093

# optimal control model vision no vision parameters
SIGMA_POS_VISION = DEFAULT_SIGMA_POS
SIGMA_POS_NO_VISION = SIGMA_POS_VISION + 0.005

# palette for masking models
_pal = sns.color_palette("colorblind", 10)
MODEL_COLORS = {
    "OFC K-gain": "black",
    "Conv vel (Asym)": _pal[2],
    "Conv acc (Asym)": _pal[4],
    "Conv v+a (Asym)": _pal[3],
}
