function test_kwave_helmholtz(in_filename, plot_tests);
% Function to generate k-Wave results compare j-Wave and k-Wave.
% Solving the 2D Helmholtz equation as a steady state of
% the wave propagation.
%
% Arguments:
% in_filename: name of the input file to setup the simulation
% plot_tests: boolean to plot the results of the tests

  arguments
    in_filename
    plot_tests = false
  end

  out_filename = strrep(in_filename, 'setup_', '');
  jw = load(in_filename);

  % Defining literals
  Nx = single(jw.Nx);
  dx = single(jw.dx);
  w0 = single(jw.omega);
  medium.sound_speed = single(jw.sound_speed);
  medium.density = single(jw.density);
  medium.alpha_coeff = single(jw.attenuation);
  medium.alpha_power = 2.0;
  source_mag = single(jw.source_magnitude);
  source_location = jw.source_location;
  pml_size = single(jw.pml_size);

  % Other settings
  kgrid = kWaveGrid(Nx(1), dx(1), Nx(2), dx(2));
  source_freq = w0/(2*pi);
  cfl = 0.1;
  roundtrips = 20;
  medium.sound_speed_ref = min(medium.sound_speed(:));
  medium.alpha_power = 2.0;

  % calculate the time step using an integer number of points per period
  ppw = (medium.sound_speed_ref/source_freq)/dx(1); % points per wavelength
  ppp = ceil(ppw / cfl);      % points per period
  T   = 1 / source_freq;             % period [s]
  dt  = T / ppp;              % time step [s]

  % calculate the number of time steps to reach steady state
  t_end = roundtrips*sqrt( kgrid.x_size.^2 + kgrid.y_size.^2 )/min(medium.sound_speed(:));
  Nt = round(t_end / dt);

  % create the time array
  kgrid.setTime(Nt, dt);

  source.p_mask = zeros(Nx(1),Nx(2));
  source.p_mask(source_location(1)+1,source_location(2)+1) = 1;

  % define the input signal
  source.p = createCWSignals(kgrid.t_array, source_freq, source_mag, 0);

  % set the sensor mask to cover the entire grid
  sensor.mask = ones(Nx(1), Nx(2));

  % record the last 3 cycles in steady state
  num_periods = 3;
  T_points = round(num_periods * T / kgrid.dt);
  sensor.record_start_index = Nt - T_points + 1;

  % input arguments
  input_args = {'CartInterp', 'nearest', 'PMLSize', pml_size, 'Smooth', false};

  % run the simulation
  sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});

  [amp, phase] = extractAmpPhase(sensor_data, 1/kgrid.dt, source_freq, ...
      'Dim', 2, 'Window', 'Rectangular', 'FFTPadding', 1);
  p_final = conj(-1j*amp.*exp(1i*phase));

  % reshape
  p_final = reshape(p_final, [Nx(1), Nx(2)]);

  % Save result
  save(out_filename, 'p_final');
