import unittest
from jwave.ode import euler_integration, semi_implicit_euler
from jax import numpy as jnp

ABS_PRECISION = 1e-6
RELATIVE_PRECISION = 1e-3


class PlanetaryOrbitsTest(unittest.TestCase):
    # inputs and outputs

    M_sun = 2.0  # kg
    p0 = jnp.array([0.0, 3.0])  # m
    v0 = jnp.array([1.0, 0.0])  # m/s
    G = 1

    dt = 0.1
    t_end = 20.0
    output_steps = (jnp.arange(0, t_end, 10 * dt) / dt).round()

    # Define equation of motion
    @staticmethod
    def newton_grav_law(G, M, r):
        """returns the applied force. Assumes Sun in (0,0)"""
        return -r * G * M / (jnp.linalg.norm(r) ** 3)

    def test_euler_integrator(self):

        # The function accepts the pair (p, v) to model the second order
        # ODE as a first order ODE (state-space representation)
        # Returns the pair (p', v')
        f = lambda x, t: (x[1], self.newton_grav_law(G=self.G, M=self.M_sun, r=x[0]))

        reference = jnp.array(
            [
                [0.991284483123007, 2.90128228144092],
                [1.92176673441599, 2.60000289891071],
                [2.74410382234362, 2.14008898376501],
                [3.43850129474233, 1.57293559860501],
                [4.00564499329517, 0.942777163452426],
                [4.45664602353859, 0.28207725332886],
                [4.80602227809836, -0.387284872154307],
                [5.06814130413606, -1.05122807095433],
                [5.25582609335828, -1.70094193839897],
                [5.38001837655501, -2.33104641382646],
                [5.44987799914836, -2.93836050023054],
                [5.47302498998557, -3.52110593830368],
                [5.45580091490605, -4.07839775032729],
                [5.40350426594792, -4.60991829671728],
                [5.3205878598367, -5.11570770550341],
                [5.21081914601735, -5.59602822848111],
                [5.07740856021777, -6.05127588013766],
                [4.92311178738908, -6.48192259060545],
                [4.75031125115653, -6.8884782394388],
            ]
        )

        p_euler, _ = euler_integration(
            f, (self.p0, self.v0), self.dt, self.output_steps, timestep_correction=1.0
        )

        assert jnp.allclose(p_euler, reference, RELATIVE_PRECISION, ABS_PRECISION)

    def test_semi_implicit_integrator(self):
        # The function accepts the pair (p, v) to model the second order
        # ODE as a first order ODE (state-space representation)
        # Returns the pair (p', v')
        f_1 = lambda v, t: v
        f_2 = lambda p, t: self.newton_grav_law(G=self.G, M=self.M_sun, r=p)

        reference = jnp.array(
            [
                [0.988081797179566, 2.90191467205812],
                [1.9096231134247, 2.60454988897942],
                [2.72013806527232, 2.15288793875843],
                [3.40190999530822, 1.59733583169729],
                [3.95624960380233, 0.980614107864983],
                [4.39396044065766, 0.334096285312753],
                [4.72894311973864, -0.320831883636564],
                [4.97498146126472, -0.970140484032244],
                [5.14446686120371, -1.60481638871757],
                [5.24807147219054, -2.21915736924293],
                [5.29481995513312, -2.80961985416929],
                [5.29229778577187, -3.37406377258468],
                [5.2468835026747, -3.91126200924486],
                [5.16396191441131, -4.42058131482259],
                [5.0481055738749, -4.90177346817943],
                [4.90322405308624, -5.35483749896856],
                [4.73268485616736, -5.77992803837576],
                [4.53941080521183, -6.17729389205345],
                [4.32595845922611, -6.54723661048397],
            ]
        )

        p_semi, _ = semi_implicit_euler(
            f_1,
            f_2,
            self.p0,
            self.v0,
            self.dt,
            self.output_steps,
            timestep_correction=1.0,
        )

        assert jnp.allclose(p_semi, reference, RELATIVE_PRECISION, ABS_PRECISION)


if __name__ == "__main__":
    unittest.main()
