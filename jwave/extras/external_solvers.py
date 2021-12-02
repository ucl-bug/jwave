from jwave.extras.engine import Matlab
import numpy as np
import matlab.engine
from typing import Union

class kWaveSolver(object):
    r"""
    This object can be used to use k-Wave in place of the `jwave` solver. 
    Mainly useful for having an external reference to compare results.
    """
    def __init__(self, kwave_path=None, offgrid_path=None):
        r"""
        Constructor for the kWaveSolver.

        Args:
            kwave_path ([type], optional): Path to the k-Wave toolbox.
                If `None`, the path is loaded from the environment variable
                `KWAVE_CORE_PATH`.
            offgrid_path ([type], optional):  Path to the Off-Grid-Sources toolbox.
                If `None`, the path is loaded from the environment variable
                `KWAVE_OFFGRID_PATH`.
        """
        self.matlab = Matlab(kwave_path, offgrid_path)
        self.start()

    def solve(
        self, 
        params: dict, 
        dx: float, 
        t_end: float, 
        sources=None, 
        sensors=None
    ) -> Union[np.ndarray, np.array]:
        r"""
        Solve the wave equation using k-Wave.

        Args:
            params (dict): Parameters for the simulation. This dictionary
                is returned by `jwave.acoustis.on_grid_wave_propagation`.
            dx (float): Spatial resolution.
            t_end (float): End time of the simulation. **Currently ignored**.
            sources (geometry.Sources, optional): Sources object. Defaults to None.
            sensors (geometry.Sensors, optional): Sensors object. Defaults to None.

        Returns:
            np.ndarray: The pressure signal at the senors locations. **Note that the
                order of the senors is not the same as `jwave`**.
            np.ndarray: The time axis.
        """
        # Collect params
        dt = params["integrator"]['dt']
        c = params["acoustic_params"]["speed_of_sound"][...,0]
        N = c.shape
        density = params["acoustic_params"]["density"][...,0]

        # Add to matlab
        self.matlab.add(N, "N")
        self.matlab.add(dx, "dx")
        self.matlab.add(dt, "dt")
        self.matlab.add(density, "density")

        # Construct grid
        if len(N) == 1:
            self.matlab.run(f"kWaveGrid({N}, {dx})", "kgrid")
        elif len(N) == 2:
            self.matlab.run(f"kWaveGrid({N[0]}, {dx[0]}, {N[1]}, {dx[1]})", "kgrid")
        elif len(N) == 3:
            self.matlab.run(f"kWaveGrid({N[0]}, {dx[0]}, {N[1]}, {dx[1]}, {N[2]}, {dx[2]})", "kgrid")
        else:
            raise ValueError(f"Grid size {N} not supported")

        # Make source and sensors into binary matrices
        sources_struct = dict()
        sensors_struct = dict()
        if sources is not None:
            src_mask = sources.to_binary_mask(N)
            sources_struct["mask"] = matlab.double(src_mask.tolist())
        else:
            src_mask = None
        if sensors is not None:
            sens_mask = sensors.to_binary_mask(N)
            sensors_struct["mask"] = matlab.double(sens_mask.tolist())
        else:
            sens_mask = np.ones(N, dtype=np.bool)
            sensors_struct["mask"] = matlab.logical(sens_mask.tolist())

        medium = dict()
        medium["sound_speed"] = matlab.double(c.tolist())
        medium["density"] = matlab.double(density.tolist())
        self.matlab.add(medium, "medium")

        # Add initial fields if not none
        sources_struct["p0"] = matlab.double(params["initial_fields"]["p"].tolist())

        # Add sources and sensors
        self.matlab.add(sources_struct, "sources")
        self.matlab.add(sensors_struct, "sensors")

        # Simulate
        sensor_data = self.matlab.run(
            'kspaceFirstOrder2D(kgrid, medium, sources, sensors);', 
            nargout=1
        )
        self.matlab.add(sensor_data, "sensor_data")
        sensor_data = self.matlab.run(
            "reorderSensorData(kgrid, sensors, sensor_data);",
            nargout=1
        )
        sensor_data = np.asarray(sensor_data, dtype="float32")

        # get time axis
        time_axis = self.matlab.run(
            "kgrid.t_array", nargout=1
        )
        time_axis = np.asarray(time_axis, dtype="float32")[0]
        return sensor_data, time_axis

    def stop(self):
        self.matlab.stop()
    
    def start(self):
        self.matlab.start()

if __name__ == "__main__":
    from jwave.geometry import Domain
    from jax import numpy as jnp
    from jwave.geometry import Medium,TimeAxis,_points_on_circle, Sensors, _circ_mask
    from jwave.acoustics import ongrid_wave_propagation

    N, dx = (128, 128), (0.1e-3, 0.1e-3)
    domain = Domain(N, dx)
    sound_speed = jnp.ones(N)*1500
    medium = Medium(domain=domain, sound_speed=sound_speed)
    time_axis = TimeAxis.from_medium(medium, cfl=0.3)

    num_sensors = 32
    x, y = _points_on_circle(num_sensors,40,(64,64))
    sensors_positions = (jnp.array(x), jnp.array(y))
    sensors = Sensors(positions=sensors_positions)

    mask1 = _circ_mask(N, 8, (50,50))
    mask2 = _circ_mask(N, 5, (80,60))
    mask3 = _circ_mask(N, 10, (64,64))
    mask4 = _circ_mask(N, 30, (64,64))
    p0 = 5.*mask1 + 3.*mask2 + 4.*mask3 + 0.5*mask4
    params, _ = ongrid_wave_propagation(
        medium=medium,
        time_array=time_axis,
        output_t_axis = time_axis,
        sensors=sensors,
        backprop=True,
        p0 = p0
    )

    solver = kWaveSolver()
    p = solver.solve(
        params, 
        dx, 
        time_axis.t_end,
        sources=None,
        sensors=sensors,
        u0=None,
        p0=p0
    )
    print(p.shape)