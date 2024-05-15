import os
import bz2
import _pickle
from glob import glob

import streamlit as st
import numpy as np

from plotly.subplots import make_subplots
from plotly import colors
import plotly.graph_objects as go

DATA_DIR = "epsc2024/out"
EXTENSION = "pbz2"
CROSS_SECTION_SCALE = 1e6
CMAP_TYPE = "Turbo"
CMAP_DELTA = 0.1
st.set_page_config(page_title="", layout="wide")

files_cbox = {}
with st.sidebar:
    form = st.form("files")
    path = f"{DATA_DIR}/**/*.{EXTENSION}"
    files = glob(path, recursive=True)
    form.write("Files")
    for i, f in enumerate(sorted(files)):
        files_cbox[f] = form.checkbox(f, value=(i == 0))
    submit = form.form_submit_button("Submit")
files = [key for key, value in files_cbox.items() if value]
if not files:
    st.warning("Please select at least one file")
    st.stop()

wavelengths = None
scattering_angles = None

scattering_cross_section = {}
extinction_cross_section = {}
single_scattering_albedo = {}
phase_function = {}
degree_of_linear_polarization = {}
degree_of_linear_polarization_q = {}
degree_of_linear_polarization_u = {}
degree_of_circular_polarization = {}

for file in files:
    data = bz2.BZ2File(file, "rb")
    data = _pickle.load(data)

    if wavelengths is None:
        wavelengths = np.array(data["wavelength"]["value"])
    else:
        np.testing.assert_allclose(
            wavelengths,
            np.array(data["wavelength"]["value"]),
            1e-9,
            0,
            err_msg="All wavelength arrays need to be the same!",
        )

    if scattering_angles is None:
        scattering_angles = np.array(data["angle"]["value"])
    else:
        np.testing.assert_allclose(
            scattering_angles,
            np.array(data["angle"]["value"]),
            1e-9,
            0,
            err_msg="All scattering angle arrays need to be the same!",
        )

    scattering_cross_section[file] = (
        data["wavelength"]["data"]["scattering_cross_section"] * CROSS_SECTION_SCALE**2
    )
    extinction_cross_section[file] = (
        data["wavelength"]["data"]["extinction_cross_section"] * CROSS_SECTION_SCALE**2
    )
    single_scattering_albedo[file] = data["wavelength"]["data"][
        "single_scattering_albedo"
    ]
    phase_function[file] = data["angle"]["data"]["phase_function"]["normal"]
    degree_of_linear_polarization[file] = data["angle"]["data"][
        "degree_of_linear_polarization"
    ]["normal"]
    degree_of_linear_polarization_q[file] = data["angle"]["data"][
        "degree_of_linear_polarization_q"
    ]["normal"]
    degree_of_linear_polarization_u[file] = data["angle"]["data"][
        "degree_of_linear_polarization_u"
    ]["normal"]
    degree_of_circular_polarization[file] = data["angle"]["data"][
        "degree_of_circular_polarization"
    ]["normal"]
scattering_angles = scattering_angles * 180 / np.pi

wavelengths_cbox = []
with st.sidebar:
    form = st.form("wavelengths")
    form.write("Wavelengths")
    for i, w in enumerate(wavelengths):
        wavelengths_cbox.append(form.checkbox(f"{w / 1e3:.2f}&mu;m", value=(i == 0)))
    submit = form.form_submit_button("Submit")
if not wavelengths_cbox:
    st.warning("Please select at least one wavelength to display")
    st.stop()

# Display wavelength stuff
cmap_delta = 0.1
cmap = colors.sample_colorscale(
    CMAP_TYPE, np.linspace(CMAP_DELTA, 1 - CMAP_DELTA, len(files))
)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
# Scattering Cross-Section
for idx, file in enumerate(files):
    name = file.split(os.sep)[-1]
    fig.add_trace(
        go.Scatter(
            x=wavelengths / 1e3,
            y=scattering_cross_section[file],
            line=dict(color=cmap[idx]),
            name="C<sub>sca</sub>",
            text=file,
            legendgrouptitle_text=name,
            legendgroup=file,
        ),
        row=1,
        col=1,
    )
    # Extinction Cross-Section
    fig.add_trace(
        go.Scatter(
            x=wavelengths / 1e3,
            y=extinction_cross_section[file],
            line=dict(color=cmap[idx]),
            name="C<sub>ext</sub>",
            legendgroup=file,
        ),
        row=2,
        col=1,
    )
    # Single-Scattering Albedo
    fig.add_trace(
        go.Scatter(
            x=wavelengths / 1e3,
            y=single_scattering_albedo[file],
            line=dict(color=cmap[idx]),
            name="w",
            legendgroup=file,
        ),
        row=3,
        col=1,
    )

fig.update_layout(
    title="Mixing components",
    height=900,
    xaxis3=dict(title="Wavelength", ticksuffix="&mu;m"),
    yaxis1=dict(
        title="Scattering Cross-section [&mu;m² ]",
        showexponent="all",
        exponentformat="e",
    ),
    yaxis2=dict(
        title="Extinction Cross-section [&mu;m²]",
        showexponent="all",
        exponentformat="e",
    ),
    yaxis3=dict(title="Single-Scattering Albedo"),
)
st.plotly_chart(fig, use_container_width=True)

# Plot angle stuff
# cmap = colors.sample_colorscale(CMAP_TYPE, np.linspace(CMAP_DELTA, 1-CMAP_DELTA, wavelength.size))
for idx, toggle in enumerate(wavelengths_cbox):
    if not toggle:
        continue

    st.header(f"λ = {wavelengths[idx]/1e3:.2f}&mu;m")
    fig = make_subplots(rows=2, cols=2)
    for f_idx, file in enumerate(files):
        name = file.split(os.sep)[-1]
        fig.add_trace(
            go.Scatter(
                x=scattering_angles,
                y=phase_function[file][:, idx],
                line=dict(color=cmap[f_idx]),
                name="Phase function",
                text=f"λ = {wavelengths[idx]}",
                legendgrouptitle_text=name,
                legendgroup=name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=scattering_angles,
                y=degree_of_linear_polarization[file][:, idx],
                line=dict(color=cmap[f_idx]),
                name="DoLP",
                text=f"λ = {wavelengths[idx]}",
                legendgroup=name,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=scattering_angles,
                y=degree_of_linear_polarization_q[file][:, idx],
                line=dict(color=cmap[f_idx]),
                name="DoLP Q",
                text=f"λ = {wavelengths[idx]}",
                legendgroup=name,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=scattering_angles,
                y=degree_of_circular_polarization[file][:, idx],
                line=dict(color=cmap[f_idx]),
                name="DoCP",
                text=f"λ = {wavelengths[idx]}",
                legendgroup=name,
            ),
            row=2,
            col=2,
        )
    fig.update_layout(
        # title="Log-plot and Polar-plot of the " + plot_type,
        height=800,
        xaxis1=dict(
            title="Phase Angle",
            ticksuffix="°",
            tickmode="linear",
            tick0=0,
            dtick=45,
        ),
        yaxis1=dict(title="Phase Function p(θ, λ)"),
        xaxis2=dict(
            title="Phase Angle",
            ticksuffix="°",
            tickmode="linear",
            tick0=0,
            dtick=45,
        ),
        yaxis2=dict(title="Degree of linear polarization (θ, λ)"),
        xaxis3=dict(
            title="Phase Angle",
            ticksuffix="°",
            tickmode="linear",
            tick0=0,
            dtick=45,
        ),
        yaxis3=dict(title="Degree of linear polarization - Q (θ, λ)"),
        xaxis4=dict(
            title="Phase Angle",
            ticksuffix="°",
            tickmode="linear",
            tick0=0,
            dtick=45,
        ),
        yaxis4=dict(title="Degree of circular polarization (θ, λ)"),
    )
    st.plotly_chart(fig, use_container_width=True)

manual_idx = [1, 6, 11]

# Phase function
fig = make_subplots(rows=1, cols=3, shared_xaxes=True, vertical_spacing=0.02)
# Scattering Cross-Section
for idx, file in enumerate(files):
    name = file.split(os.sep)[-1]
    for i, m in enumerate(manual_idx):
        fig.add_trace(
            go.Scatter(
                x=scattering_angles,
                y=phase_function[file][:, m],
                line=dict(color=cmap[idx]),
                name=f"λ = {wavelengths[m] / 1e3}&mu;m",
                legendgrouptitle_text=name,
                legendgroup=name,
            ),
            row=1,
            col=i + 1,
        )

fig.update_layout(
    # title="Mixing components",
    height=650,
    xaxis1=dict(
        title="Phase Angle",
        ticksuffix="°",
        tickmode="linear",
        tick0=0,
        dtick=45,
    ),
    yaxis1=dict(
        title=f"Phase Function p(θ, λ = {wavelengths[manual_idx[0]] / 1e3}&mu;m)"
    ),
    xaxis2=dict(
        title="Phase Angle",
        ticksuffix="°",
        tickmode="linear",
        tick0=0,
        dtick=45,
    ),
    yaxis2=dict(
        title=f"Phase Function p(θ, λ = {wavelengths[manual_idx[1]] / 1e3}&mu;m)"
    ),
    xaxis3=dict(
        title="Phase Angle",
        ticksuffix="°",
        tickmode="linear",
        tick0=0,
        dtick=45,
    ),
    yaxis3=dict(
        title=f"Phase Function p(θ, λ = {wavelengths[manual_idx[2]] / 1e3}&mu;m)"
    ),
)
st.plotly_chart(fig, use_container_width=True)

# DoLP
fig = make_subplots(rows=1, cols=3, shared_xaxes=True, vertical_spacing=0.02)
# Scattering Cross-Section
for idx, file in enumerate(files):
    name = file.split(os.sep)[-1]
    for i, m in enumerate(manual_idx):
        fig.add_trace(
            go.Scatter(
                x=scattering_angles,
                y=degree_of_linear_polarization[file][:, m],
                line=dict(color=cmap[idx]),
                name=f"λ = {wavelengths[m] / 1e3}&mu;m",
                legendgrouptitle_text=name,
                legendgroup=name,
            ),
            row=1,
            col=i + 1,
        )

fig.update_layout(
    # title="Mixing components",
    height=650,
    xaxis1=dict(
        title="Phase Angle",
        ticksuffix="°",
        tickmode="linear",
        tick0=0,
        dtick=45,
    ),
    yaxis1=dict(title=f"DoLP(θ, λ = {wavelengths[manual_idx[0]] / 1e3}&mu;m)"),
    xaxis2=dict(
        title="Phase Angle",
        ticksuffix="°",
        tickmode="linear",
        tick0=0,
        dtick=45,
    ),
    yaxis2=dict(title=f"DoLP(θ, λ = {wavelengths[manual_idx[1]] / 1e3}&mu;m)"),
    xaxis3=dict(
        title="Phase Angle",
        ticksuffix="°",
        tickmode="linear",
        tick0=0,
        dtick=45,
    ),
    yaxis3=dict(title=f"DoLP(θ, λ = {wavelengths[manual_idx[2]] / 1e3}&mu;m)"),
)
st.plotly_chart(fig, use_container_width=True)
