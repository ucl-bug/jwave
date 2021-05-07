% Script to define a geometry to run in JAX
%
% author: Bradley Treeby
% date: 13 February 2021
% last update: 5 March 2021 (Antonio Stanziola)

% add other toolboxes to the path
addpath('/mnt/Software/k-Plan/toolbox/');
addpath('/mnt/Software/k-Plan/libraries/nifti/');

% =========================================================================
% LITERALS
% =========================================================================

% source properties
source_f0       = 200e3;    % [Hz]
source_mag      = 40e3;     % [Pa]
bowl_roc        = 64e-3;    % bowl radius of curvature [m]
bowl_diameter   = 64e-3;    % bowl aperture diameter [m]

% source position
source_pos      = [-0.07, -0.08];     % [m]
focus_pos       = [0.025, 0.03];           % [m]

% image properties
img_slice_idx   = 80;   
img_scale       = 1;

% =========================================================================
% LOAD SKULL AND SETUP GRID
% =========================================================================

% load the CT data (requires the nifti toolbox to be on the path)
nii = load_nii(ct_location);

% choose a slice
img_slice = single(nii.img(:, :, img_slice_idx));

% force to be even
if rem(size(img_slice, 1), 2)
    img_slice(end, :) = [];
end
if rem(size(img_slice, 2), 2)
    img_slice(:, end) = [];
end

% pad and resample
img_pad = [(256 - size(img_slice, 1))/2, (256 - size(img_slice, 2))/2];
img_slice = expandMatrix(img_slice, img_pad, inf);
img_slice(img_slice == inf) = -1024;
img_slice = img_slice(1:2:end, 1:2:end);
dx = 1e-3 * nii.hdr.dime.pixdim(2) * 2;
dy = 1e-3 * nii.hdr.dime.pixdim(3) * 2;


% img_slice = resize(img_slice, size(img_slice) * img_scale);

% get the grid size
[Nx, Ny] = size(img_slice);
% dx = 1e-3 * nii.hdr.dime.pixdim(2) / img_scale;
% dy = 1e-3 * nii.hdr.dime.pixdim(3) / img_scale;

% convert to acoustic properties
[medium, skull_mask] = skull2medium(img_slice, [], source_f0, 'IncludeAir', false, 'ReturnValues', 'acoustic-linear');

% take the skull mask, fill, and erode to give a brain mask
brain_mask = bwfill(skull_mask, 'holes');
brain_mask = imerode(brain_mask, strel('disk', 8));
brain_mask = single(brain_mask);
skull_mask = single(skull_mask);

% figure;
% imagesc(brain_mask + 2 * skull_mask);

% structure arrays are not supported by the interface, so give new names
sound_speed = medium.sound_speed;
attenuation = medium.alpha_coeff;

% frequency in radians
omega = 2 * pi * source_f0;

% =========================================================================
% DEFINE SOURCE
% =========================================================================

% create empty kWaveArray
karray = kWaveArray('BLITolerance', 0.1);

% add arc shaped element
karray.addArcElement(source_pos, bowl_roc, bowl_diameter, focus_pos);

% get weighted source mask
kgrid = kWaveGrid(Nx, dx, Ny, dy);
p_mask = complex(karray.getArrayGridWeights(kgrid));

% % plot grid weights
% figure;
% imagesc(kgrid.y_vec - kgrid.y_vec(1), kgrid.x_vec - kgrid.x_vec(1), single(karray.getArrayBinaryMask(kgrid) | skull_mask));
% axis image;
% title('Off-grid mask and skull mask');

% display output
disp('MATLAB script executed successfully');
