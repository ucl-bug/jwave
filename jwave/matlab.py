import matlab.engine
import os
import io
from functools import partial


class Matlab(object):
    r"""
    A bridge object that allows python to interact with a MATLAB session
    """
    def __init__(self, kwave_path=None, offgrid_path=None):
        """Constructs a new Matlab object.

        Args:
            kwave_path ([type], optional): Path to the k-Wave toolbox.
                If `None`, the path is loaded from the environment variable
                `KWAVE_CORE_PATH`.
            offgrid_path ([type], optional):  Path to the Off-Grid-Sources toolbox.
                If `None`, the path is loaded from the environment variable
                `KWAVE_OFFGRID_PATH`.

        [^1]: [http://www.k-wave.org/](http://www.k-wave.org/)

        [^2]: Elliott S. Wise et al., 2019. [*Representing arbitrary acoustic source 
            and sensor distributionsin Fourier collocation methods*](https://bug.medphys.ucl.ac.uk/papers/2019-Wise-JASA.pdf)

        !!! example
            ```python
            from jwave.matlab import Matlab
            mlb = Matlab()
            ```
        """        
        # Init variables
        self._engine = None
        if offgrid_path is None:
            self.kwave_path = os.environ.get('KWAVE_CORE_PATH')
        if offgrid_path is None:
            self.offgrid_path = os.environ.get('KWAVE_OFFGRID_PATH')

        # Redirect outputs
        self.out = io.StringIO()
        self.err = io.StringIO()

    def start(self, join_session=True):
        """Starts a MATLAB engine session.

        Args:
            join_session (bool, optional): If True, it tries to connect to
                a running instance of MATLAB. If that fails or `joint_session=False`,
                starts a new MATLAB Session
        """        
        # If a shared matlab istance is running, use it
        open_matlabs = matlab.engine.find_matlab()
        if len(open_matlabs) > 0:
            print("Connecting to Matlab session {}".format(open_matlabs[0]))
            self._engine =  matlab.engine.connect_matlab(open_matlabs[0])
        else:
            print("Opening new matlab session")
            self._engine = matlab.engine.start_matlab()

        # Add paths
        self._engine.addpath(self.kwave_path)
        self._engine.addpath(self.offgrid_path)

    def run_script(self, script_path: str):
        """Runs a MATLAB script

        Args:
            script_path (str): Script path.
        """              
        self._engine.eval("run('"+script_path+"');", nargout=0)

    def run(self, command: str, to=None, nargout=0):
        """Runs a MATLAB command in the attached session.

        Args:
            command (str): Command to be run.
            to (str, optional): If not `None`, saves the output of the command
                in the MATLAB workspace variable with the given string as name. 
            nargout (int, optional): If not `0`, returns the outputs of the function.

        Raises:
            RuntimeError: If the MATLAB engine is not running. Remember to
                start the MATLAB session using the `start()` method before
                executing MATLAB functions.

        Returns:
            [Any]: Command output
        """        
        
        if self._engine is None:
            raise RuntimeError("Matlab engine is not running, please use `start()` method")

        if to is None:
            return self._engine.eval(command, nargout=nargout, stdout=self.out,stderr=self.err)
        else:
            self._engine.workspace[to] = self._engine.eval(command, stdout=self.out,stderr=self.err)

    def add(self, value, variable_name: str):
        """Adds the `value` array to the MATLAB workspace

        Args:
            value (Numeric): Array of number to save
            variable_name (str): Name of the variable to be created in MATLAB.
        """        
        if isinstance(value, list):         # List transformed to matlab array
            value = matlab.double(value)
        self._engine.workspace[variable_name]=value

    def get(self, variable_name:str):
        """Gets a variable from the MATLAB workspace.

        Args:
            variable_name (str): Variable name in MATLAB.

        Returns:
            [type]: [description]
        """        
        return self._engine.workspace[variable_name]

    def stop(self):
        """Stops the MATLAB engine and closes the MATLAB Session.
        """        
        if self._engine is not None:
            self._engine.quit()
            self._engine = None