import datetime
import requests

import numpy as np
import streamlit as st
from astropy.io import fits
from bs4 import BeautifulSoup

st.set_page_config(layout='wide')
url = 'https://web.bv.e-technik.tu-dortmund.de/conferences/2023/epsc/'

@st.cache_resource()
def fetch_fits_from_server(url):
    data = fits.open(url)
    return data

with st.sidebar:

    form = st.form('options')

    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    files = [url + node.get('href') for node in soup.find_all('a', href=True) if node.get('href').endswith('.fits')]

    file_path = form.selectbox('Choose a polarimetry file:', sorted(files), 0, lambda x: x.split('/')[-1])
    data = fetch_fits_from_server(file_path)
    for i in range(len(data)):
        if (i > 0) and (data[i].header['XTENSION'] == 'BINTABLE'):
            continue
        data[i].data[data[i].data == 0] = 0 if isinstance(data[i].data.ravel()[0], np.integer) else np.nan
    latitude  = data['intensity'].header['latitude']
    longitude = data['intensity'].header['longitude']
    phase_angle = data['intensity'].header['S-T-O']
    timestamp = datetime.datetime.strptime(data['intensity'].header['timestamp'], '%Y-%m-%d %H:%M:%S%z')

    wavelenghts_pol = np.array([x[0] for x in data['wavelengths'].data])
    wavelenghts_pol_select = form.selectbox('Select wavelength', wavelenghts_pol, wavelenghts_pol.size-1, lambda x: f'{x:.2f} μm')
    wavelenghts_pol_idx = np.where(wavelenghts_pol == wavelenghts_pol_select)[0][0]

    percentile = form.checkbox('Percentiles of data', True, help='Only affects the DoLP and AoLP data displayed!')
    mask_slope = form.checkbox('Display only "full" channel data', True, help='Global invalid channels are filtered for first')
    # ref_or_alb = form.selectbox('Reflectance or Albedo?', ['reflectance', 'albedo'], 0, lambda x: x.title())
    ref_or_alb = 'reflectance'

    submitted = form.form_submit_button('Submit Changes')


st.write('Date:', timestamp.date(), ' - Time:', timestamp.time(), ' - Time zone:', timestamp.tzinfo)
st.write('Latitude:', latitude, ' - Longitude:', longitude)
st.write('Phase angle:', phase_angle, '°')
st.write('Region: ', data['intensity'].header['region'].title())

wac         = data['primary'].data
intensity   = data['intensity'].data[wavelenghts_pol_idx, :, :]
comparisson = data['albedo'].data[wavelenghts_pol_idx, :, :] if (ref_or_alb == 'albedo') else data['reflectance'].data[wavelenghts_pol_idx, :, :]
dolp        = data['dolp'].data[wavelenghts_pol_idx, :, :]
aolp        = data['aolp'].data[wavelenghts_pol_idx, :, :]
slope       = data['albedo_slope'].data                      if (ref_or_alb == 'albedo') else data['reflectance_slope'].data
intercept   = data['albedo_intercept'].data                  if (ref_or_alb == 'albedo') else data['reflectance_intercept'].data
clusters    = data['clusters'].data
grain_size  = data['grain_size'].data[wavelenghts_pol_idx, :, :]

if mask_slope:
    # Determine the number of invalid channels for each pixel
    mask = np.maximum(np.sum(np.isnan(data['albedo'].data), axis=0), np.sum(np.isnan(data['dolp'].data), axis=0))
    # If the number of invalid channels is greater than the number of global invalid channels, then the pixel is set to nan
    mask = mask > np.sum(np.all(np.isnan(data['albedo'].data), axis=(1,2)) | np.all(np.isnan(data['dolp'].data), axis=(1,2)))
    slope[mask]     = np.nan
    intercept[mask] = np.nan


if percentile:

    comparisson = np.clip(
        comparisson,
        np.nanpercentile(comparisson, 1),
        np.nanpercentile(comparisson, 99),
    )
    dolp = np.clip(
        dolp,
        np.nanpercentile(dolp, 1),
        np.nanpercentile(dolp, 99),
    )
    aolp = np.clip(
        aolp,
        np.nanpercentile(aolp, 1),
        np.nanpercentile(aolp, 99),
    )
    slope = np.clip(
        slope,
        np.nanpercentile(slope, 1),
        np.nanpercentile(slope, 95),
    )
    intercept = np.clip(
        intercept,
        np.nanpercentile(intercept, 1),
        np.nanpercentile(intercept, 99),
    )

    grain_size = np.clip(
        grain_size,
        np.nanpercentile(grain_size, 1),
        np.nanpercentile(grain_size, 99),
    )

    # dolp[dolp < np.nanpercentile(dolp, 1)]  = np.nan
    # dolp[dolp > np.nanpercentile(dolp, 99)] = np.nan
    # aolp[aolp < np.nanpercentile(aolp, 1)]  = np.nan
    # aolp[aolp > np.nanpercentile(aolp, 99)] = np.nan

    # slope[slope < np.nanpercentile(slope, 1)]  = np.nan
    # slope[slope > np.nanpercentile(slope, 99)] = np.nan

wac         = np.rot90(wac,         2)
intensity   = np.rot90(intensity,   2)
comparisson = np.rot90(comparisson, 2)
dolp        = np.rot90(dolp,        2)
aolp        = np.rot90(aolp,        2)
slope       = np.rot90(slope,       2)
intercept   = np.rot90(intercept,   2)
clusters    = np.rot90(clusters,    2)
grain_size  = np.rot90(grain_size,  2)

intensity = intensity.astype(float)
intensity[intensity < 1e-12] = np.nan

import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
rows = 3
cols = 3
fig, axs = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
fig.set_figheight(10)
fig.set_figwidth(10)
axs[0,0].imshow(wac, cmap='gray')
axs[0,0].set_title('WAC')
axs[0,1].imshow(intensity, cmap='jet')
axs[0,1].set_title('Intensity')
axs[0,2].imshow(comparisson, cmap='jet')
axs[0,2].set_title(ref_or_alb.title())

axs[1,0].imshow(clusters, cmap='jet')
axs[1,0].set_title('SOM')
axs[1,1].imshow(dolp, cmap='jet')
axs[1,1].set_title('DoLP')
axs[1,2].imshow(slope, cmap='jet')
axs[1,2].set_title('Slope')

axs[2,0].imshow(grain_size, cmap='jet')
axs[2,0].set_title('Rel. Grain Size')
axs[2,1].imshow(aolp, cmap='jet')
axs[2,1].set_title('AoLP')
axs[2,2].imshow(intercept, cmap='jet')
axs[2,2].set_title('Intercept')

for i in range(rows):
    for j in range(cols):
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

plt.tight_layout()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=1050, scrolling=True)
st.info(':arrow_up: **Note:** hover to the bottom left corner for zoom and pan tools.')