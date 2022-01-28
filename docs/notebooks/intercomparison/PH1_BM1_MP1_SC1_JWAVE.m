%%% Settings
ppw = 4.;     %number of points per wavelength at c0
pml_size = 10;

% source parameters
source_f0       = 500e3;   % source frequency [Hz]
source_mag      = 60;       % source pressure [kPa]
bowl_roc        = 64e-3;    % bowl radius of curvature [m]
bowl_diameter   = 64e-3;    % bowl aperture diameter [m]

% grid parameters
axial_size      = 120e-3 ;  % total grid size in the axial dimension [m]
lateral_size    = 70e-3;    % total grid size in the lateral dimension [m]
source_x_offset = 40;
source_y_offset = 40;

% off grid source parameters
bli_tolerance   = 0.02;
upsampling_rate = 10;

% medium properties
c0 = 1500;       % reference sound speed used for calculations
rho0 = 1000;     % reference density used for calculations
     
% calculate the grid spacing based on PPW and F0
dx = c0 / (ppw * source_f0);   % [m]

Nx = 256;
Ny = 196;

kgrid = kWaveGrid(Nx, dx, Ny, dx, Ny, dx);
karray = kWaveArray('BLITolerance', bli_tolerance, 'UpsamplingRate', upsampling_rate);

source_pos = [kgrid.x_vec(1) + source_x_offset * kgrid.dx(1), 0, 0];
focus_pos = [0.064 + source_x_offset * kgrid.dx(1), 0, 0];
karray.addBowlElement(source_pos, bowl_roc, bowl_diameter, focus_pos);

src_field = karray.getArrayGridWeights(kgrid);
src_field = src_field .* source_mag .* 2 ./ (dx .* c0);