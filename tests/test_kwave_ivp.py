# %% Setup simulation
from jwave.geometry import Domain, Medium, TimeAxis
from jwave.geometry import _circ_mask
from jwave.acoustics import simulate_wave_propagation

from jwave import FourierSeries

from jax import jit
from jax import numpy as jnp

Nx = (128, 128)
dx = (0.1e-3, 0.1e-3)
domain = Domain(Nx, dx)

sound_speed = 1500.0

p0 = 5.0 * _circ_mask(Nx, 5, (40, 40))
p0 =  jnp.expand_dims(p0, -1)
p0 = FourierSeries(p0, domain)

def test_ivp(use_plots = False, save_outputs = False):
    
  for test_ind in range(4):
    
    match test_ind:
      case 0:
        pml_size = 0
        smooth_initial = False
      case 1:
        pml_size = 0
        smooth_initial = True
      case 2:
        pml_size = 10
        smooth_initial = False
      case 3:
        pml_size = 10
        smooth_initial = True
        
    medium = Medium(domain = domain, sound_speed = sound_speed, pml_size=pml_size)
    time_axis = TimeAxis.from_medium(medium, cfl=0.5, t_end=5e-6)
      
    @jit
    def run_simulation(p0):
      return simulate_wave_propagation(medium, time_axis, p0=p0, smooth_initial=smooth_initial)
      
    acoustic_field = run_simulation(p0)
    p_final = acoustic_field[-1].on_grid[:,:,1]
      
    if save_outputs:
          
      from scipy.io import savemat
          
      mdic = {"p_final": p_final, "p0": p0.on_grid, "Nx": Nx, "dx": dx, 
              "Nt": time_axis.Nt, "dt": time_axis.dt, "sound_speed": sound_speed,
              "Smooth": smooth_initial, "PMLSize": pml_size}
      savemat("test_kwave_ivp_jwave_results_" + str(test_ind) + ".mat", mdic)
          
    else:
          
      from scipy.io import loadmat
         
      kwave = loadmat("test_kwave_ivp_kwave_results_" + str(test_ind) + ".mat")
      kwave_p_final = kwave["p_final"]
      err = abs(p_final - kwave_p_final)
          
      maxErr = jnp.amax(err)
      print('Maximum error = ', maxErr)
      # assert maxErr < 1e-5
          
      if use_plots:
         plot_comparison(p_final, kwave_p_final)
         
            
def plot_comparison(jwave, kwave):
    
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  import matplotlib.pyplot as plt
  
  plt.rcParams.update({'font.size': 6})
  plt.rcParams["figure.dpi"] = 300
    
  f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

  im1 = ax1.imshow(jwave)
  ax1.set_title('j-Wave')
  divider1 = make_axes_locatable(ax1)
  cax1 = divider1.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im1, cax=cax1)

  im2 = ax2.imshow(kwave)
  ax2.set_title('k-Wave')
  divider2 = make_axes_locatable(ax2)
  cax2 = divider2.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im2, cax=cax2)

  im3 = ax3.imshow((jwave - kwave))
  ax3.set_title('Difference')
  divider3 = make_axes_locatable(ax3)
  cax3 = divider3.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im3, cax=cax3)

  plt.show()
    

if __name__ == "__main__":
  test_ivp(use_plots = True, save_outputs = False)
    