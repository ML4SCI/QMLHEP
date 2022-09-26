import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm


def plot_state_overlap(overlap,epochs):
  """
  Function for plotting the state overlap between two states

  Arguments: overlap(list of overlap values at each train step)
             epochs(list with range=number of train steps)
  
  """
  fig = plt.figure(figsize=(16,9))
  gs = gridspec.GridSpec(ncols=8, nrows=8, figure=fig)
  epochs = [i for i in range(epochs)]
  epoch = epochs[-1]
  ax_loss = plt.subplot(gs[:,:4])
  ax_loss.set_xlim(0, 1.1*epoch)
  ax_loss.plot(epochs, overlap, label="State Overlap")
  ax_loss.set_xlabel('Epoch', fontsize=20)
  ax_loss.set_ylabel('Overlap', fontsize=20)
  ax_loss.grid(True)
  ax_loss.legend(fontsize=15)

#reference: https://github.com/hep-lbdl/adversarial-jets/blob/master/analysis/plots.ipynb
def plot_log_scale_image(content, vmin=1e-6, vmax=300, title=''):
  fig, ax = plt.subplots(figsize=(7, 6))
  extent=[-1.25, 1.25, -1.25, 1.25]
  im = ax.imshow(content, interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent)
  cbar = plt.colorbar(im, fraction=0.05, pad=0.05)
  cbar.set_label(r'Pixel $p_T$ (GeV)', y=0.85)
  plt.xlabel(r'[Transformed] Pseudorapidity $(\eta)$')
  plt.ylabel(r'[Transformed] Azimuthal Angle $(\phi)$')
  plt.title(title)