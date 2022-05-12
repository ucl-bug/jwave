function test_kwave_ivp(plot_tests)
% Function to generate k-Wave results compare j-Wave and k-Wave. The k-Wave simulation is run for
% one 
% extra time step, as j-Wave assigns p0 at the beginning of the time loop,
% and k-Wave at the end.

arguments
    plot_tests = false
end

numTests = 4;

for testInd = 0:(numTests - 1)

    in_filename = ['test_kwave_ivp_jwave_results_' num2str(testInd) '.mat'];
    out_filename = strrep(in_filename, 'jwave', 'kwave');

    jw = load(in_filename);
    jw.Nx = single(jw.Nx);
    
    kgrid = kWaveGrid(jw.Nx(1), jw.dx(1), jw.Nx(2), jw.dx(2));
    kgrid.setTime(jw.Nt + 1, jw.dt);
    
    medium.sound_speed = jw.sound_speed;
    source.p0 = jw.p0;
    sensor.record = {'p_final'};
    
    kw = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
        'PMLSize', double(jw.PMLSize), ...
        'PlotSim', false, ...
        'Smooth', jw.Smooth);
    
    if plot_tests
        compare2D(jw.p_final, kw.p_final);
    end
    
    p_final = kw.p_final;
    save(out_filename, 'p_final');

end

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