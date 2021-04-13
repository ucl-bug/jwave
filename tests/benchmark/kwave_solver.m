function [p, exec_time] = kwave_solution(...
    sos_map, ...
    dx, ...
    source_location, ...
    source_signal, ...
    dt, ...
    t_max);
    % Solving the 2D Helmholtz equation as a steady state of
    % the wave propagation using kwave

    % set the literals
    L = size(sos_map,1);
    Nx = double(L);           % number of grid points in the x (row) direction
    Ny = double(L);           % number of grid points in the y (column) direction
    dx = double(dx);
    dy = dx;                  % grid point spacing in the y direction [m]
    sos_map = squeeze(sos_map);
    dt=double(dt);
    source_location=double(source_location);
    source_signal=double(source_signal);
    
    disp([Nx,dx,Ny,dy])
    kgrid = kWaveGrid(Nx, dx, Ny, dy);
    
    % define the properties of the propagation medium
    medium.sound_speed = sos_map;	% [m/s]
    medium.sound_speed_ref = min(sos_map(:));
    medium.density = 1.;
    
    % calculate the number of time steps to reach steady state
    Nt = round(t_max / dt);
    
    % create the time array
    kgrid.setTime(Nt, dt);
    
    source.p_mask = zeros(L);
    source.p_mask(source_location(1)+1,source_location(2)+1) = 1;
    
    % define the input signal
    source.p = source_signal;
    
    sensor.mask = ones(Nx, Ny);
    sensor.record_start_index = Nt - 2;

    % input arguments
    input_args = {'CartInterp', 'nearest', 'PMLSize', 30, 'DataPath','/tmp/', 'DataName','jwave', 'DeleteData', false};

    % run the simulation
    sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});

    p=sensor_data(:,end);

    % Get exec time
    exec_time = h5readatt(['/tmp/', 'jwave_output.h5'], '/', 'simulation_phase_execution_time');
    exec_time = exec_time(~isspace(exec_time));
    exec_time(exec_time == 's') = [];
    exec_time = str2num(exec_time);
    end
