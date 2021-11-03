% This scripts generates the reference solution for a homogeneous
% wave propagation medium.
%
% The code is adapted from 
% http://www.k-wave.org/documentation/example_ivp_homogeneous_medium.php
%
% WARNING: Ensure that k-Wave is installed correctly and in path
%          before running this script.
clearvars;

% Adding k-Wave path
addpath(genpath('/mnt/Software/k-Wave'));
addpath(genpath('~/repos/k-plan-qms-sem/toolbox'))

% create the computational grid
Nx = 128;           % number of grid points in the x (row) direction
Ny = 128;           % number of grid points in the y (column) direction
dx = 0.1e-3;        % grid point spacing in the x direction [m]
dy = 0.1e-3;        % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% define the properties of the propagation medium
medium.sound_speed = 1500;

% create initial pressure distribution using makeDisc
disc_magnitude = 5; % [Pa]
disc_x_pos = 50;    % [grid points]
disc_y_pos = 50;    % [grid points]
disc_radius = 8;    % [grid points]
disc_1 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

disc_magnitude = 3; % [Pa]
disc_x_pos = 80;    % [grid points]
disc_y_pos = 60;    % [grid points]
disc_radius = 5;    % [grid points]
disc_2 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

source.p0 = disc_1 + disc_2;

% place sensors everywhere
sensor.mask = ones(Nx, Ny);

% run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, 'Smooth', false);

% Reshape sensor data to Nx x Ny x t
sensor_data = reshape(sensor_data, Nx, Ny, []);

% Save image
%{
imagesc(squeeze(sensor_data(:,:,100)));
saveas(gcf, 'test.png','png');
close all
%}

% Save results and setup
results.N = [Nx, Ny];
results.dx = [dx, dy];
results.dt = kgrid.dt;
results.p0 = source.p0;
results.sensor_data = sensor_data;

save('homogeneous_ref.mat', 'results');

