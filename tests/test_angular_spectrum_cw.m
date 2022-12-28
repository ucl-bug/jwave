function test_angular_spectrum_cw(in_filename, plot_tests)
  % Function to generate k-Wave results compare j-Wave and k-Wave. The k-Wave
  % simulation is run for one extra time step, as j-Wave assigns p0 at the
  % beginning of the time loop, and k-Wave at the end.
  %
  % Arguments:
  % in_filename: name of the input file to setup the simulation
  % plot_tests: boolean to plot the results of the tests

  % This file is part of j-Wave.
  %
  % j-Wave is free software: you can redistribute it and/or
  % modify it under the terms of the GNU Lesser General Public
  % License as published by the Free Software Foundation, either
  % version 3 of the License, or (at your option) any later version.
  %
  % j-Wave is distributed in the hope that it will be useful, but
  % WITHOUT ANY WARRANTY; without even the implied warranty of
  % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  % Lesser General Public License for more details.
  %
  % You should have received a copy of the GNU Lesser General Public
  % License along with j-Wave. If not, see <https://www.gnu.org/licenses/>.

    arguments
        in_filename
        plot_tests = false
    end
    
    % Add to path k-wave
    addpath(getenv('KWAVE_CORE_PATH'));

    % Load file
    out_filename = strrep(in_filename, 'setup_', '');
    jw = load(in_filename);

    % Setup inputs
    input_plane = single(jw.pressure);
    dx = single(jw.dx);
    z_pos = single(jw.z_pos);
    f0 = single(jw.f0);
    c0 = single(jw.c0);
    angular_restiction = logical(jw.angular_restriction);
    padding = int32(jw.padding);

    medium.sound_speed = c0;

    fft_length = size(input_plane, 1);

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
