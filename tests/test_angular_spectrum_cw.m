function test_angular_spectrum_cw(in_filename, plot_tests)
  % Function to generate k-Wave results compare j-Wave and k-Wave. The k-Wave
  % simulation is run for one extra time step, as j-Wave assigns p0 at the
  % beginning of the time loop, and k-Wave at the end.
  %
  % Arguments:
  % in_filename: name of the input file to setup the simulation
  % plot_tests: boolean to plot the results of the tests

    arguments
        in_filename
        plot_tests = false
    end

    % Load file
    out_filename = strrep(in_filename, 'setup_', '');
    jw = load(in_filename);

    % Setup inputs
    input_plane = single(jw.pressure);
    dx = single(jw.dx);
    z_pos = single(jw.z_pos);
    f0 = single(jw.f0);
    c0 = single(jw.c0);
    alpha = single(jw.alpha);
    alpha_power = single(jw.alpha_power);
    angular_restiction = logical(jw.angular_restriction);
    padding = int32(jw.padding);

    medium.sound_speed = c0;
    medium.alpha_power = alpha_power;
    medium.alpha_coeff = alpha;

    % Run solver
    p_plane = angularSpectrumCW(...
      input_plane, ...
      dx, ...
      z_pos, ...
      f0, ...
      medium, ...
      "AngularRestriction", angular_restiction, ...
      "GridExpansion", padding);

    % Save results
    save(out_filename, 'p_plane');
