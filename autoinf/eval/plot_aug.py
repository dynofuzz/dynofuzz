import json
import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython import embed

inst_2_num_records_original = json.load(open("inst_2_num_records_from_collection.json"))
inst_2_num_records_after_aug = json.load(open("inst_2_num_records_after_aug.json"))

original = np.array(list(inst_2_num_records_original.values()))
after_aug = np.array(list(inst_2_num_records_after_aug.values()))
print(f"{min(original) = }, {max(original) = }")
print(f"{min(after_aug) = }, {max(after_aug) = }")

bins = 100_000
rang = (0, 2600)
delta = rang[1] / bins
density = True
prob_original, edges = np.histogram(original, bins=bins, range=rang, density=density)
prob_after, _ = np.histogram(after_aug, bins=bins, range=rang, density=density)
prob_original *= delta
prob_after *= delta
middles = (edges[:-1] + edges[1:]) / 2
# middles = np.log10(middles + 1)
cdf_original = np.cumsum(prob_original)
cdf_after = np.cumsum(prob_after)

print(sum(prob_original))
print(sum(prob_after))


def latex_text(s: str) -> str:
    return repr(("$\text{" + s + "}$"))[1:-1]


fig = go.Figure()
# fig.add_trace(go.Histogram(
#     x=X,
#     name=latex_text('before augmentation'),
# ))
# fig.add_trace(go.Histogram(
#     x=Y,
#     name=latex_text('after augmentation'),
# ))
fig.add_traces(
    [
        go.Scatter(
            x=middles,
            y=cdf_original,
            mode="lines",
            name=latex_text("before augmentation"),
        ),
        go.Scatter(
            x=middles, y=cdf_after, mode="lines", name=latex_text("after augmentation")
        ),
    ]
)

fig.update_xaxes(
    tickprefix=r"$",
    ticksuffix=r"$",
    showgrid=False,
    showline=True,
    mirror=True,
    linewidth=1,
    linecolor="black",
    ticks="inside",
    rangemode="tozero",
)
fig.update_yaxes(
    tickprefix=r"$",
    ticksuffix=r"$",
    showgrid=False,
    showline=True,
    mirror=True,
    linewidth=1,
    linecolor="black",
    ticks="inside",
    rangemode="normal",
)
fig.update_layout(
    title={
        "text": latex_text("Distribution of #records over operator instances"),
        "x": 0.5,
    },
    # xaxis_title=r"$\log_{10}(\text{#records} + 1)$",
    xaxis_title=latex_text("#Records"),
    yaxis_title=latex_text("Proportion of operator instances"),
    # yaxis={ "tickprefix": r"$" , "ticksuffix": r"$", },
    legend=dict(
        yanchor="top",
        y=0.80,
        xanchor="right",
        x=0.99,
    ),
    margin=dict(l=0, r=10, b=0, t=30, pad=0),
    # margin_pad=10,
    font=dict(size=16, color="black"),
    plot_bgcolor="rgba(0, 0, 0, 0)",
)

# Overlay both histograms
# fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
# fig.update_traces(opacity=0.75)
# fig.show()
# fig.write_html('inst_in_records.html')
fig.write_image("dist_records.png", width=800, height=600)
fig.write_image("dist_records.pdf", width=800, height=600)
