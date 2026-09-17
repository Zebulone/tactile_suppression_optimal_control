import pandas as pd

from suppression_oc.util import signal_detection_parameters

detection_data = pd.read_csv("./data/tatai_et_al_detection.csv")

signal_detection_res = detection_data.groupby(["subject", "condition"]).apply(
    signal_detection_parameters, include_groups=False
)

signal_detection = (
    signal_detection_res.apply(lambda x: x[0])
    .stack(level=0)
    .reset_index()
    .rename(columns={0: "d_prime"})
)
signal_detection["criterion"] = (
    signal_detection_res.apply(lambda x: x[1]).stack(level=0).reset_index()[0]
)

signal_detection.to_csv(
    "./data/signal_detection.csv",
    index=False,
)
