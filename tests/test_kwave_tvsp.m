function test_kwave_tvsp(in_filename, plot_tests)
% Function to generate k-Wave results compare j-Wave and k-Wave.
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
    jw.Nx = single(jw.Nx);
    
    kgrid = kWaveGrid(jw.Nx(1), jw.dx(1), jw.Nx(2), jw.dx(2));
    kgrid.setTime(jw.Nt, jw.dt);
    
    medium.sound_speed = jw.sound_speed;
    medium.density = jw.density;
    
    source.p_mask = zeros(kgrid.Nx, kgrid.Ny);
    for ind = 1:size(jw.source_positions, 2)
        source.p_mask(jw.source_positions(1, ind), jw.source_positions(2, ind)) = 1;
    end
    source.p = jw.source_signals;
    source.p_mode = 'additive-no-correction';
    
    sensor.record = {'p_final'};
    
    kw = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
      'PMLSize', double(jw.PMLSize), ...
      'PlotSim', false);
    
    if plot_tests
        compare2D(jw.p_final, kw.p_final);
    end

    p_final = kw.p_final;
    save(out_filename, 'p_final');


function compare2D(jwave, kwave)

    figure;
    subplot(1, 3, 1);
    imagesc(jwave);
    axis image;
    colorbar;
    title('j-Wave');
    
    subplot(1, 3, 2);
    imagesc(kwave);
    axis image;
    colorbar;
    title('k-Wave');
    
    subplot(1, 3, 3);
    imagesc(jwave - kwave);
    axis image;
    colorbar;
    title('Difference');
    
    scaleFig(2, 1);
