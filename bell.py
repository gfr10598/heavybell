"""
Heavy bell computes a smooth force to be applied to the rope to achieve
a given interval and energy.  It controls the trajectory of the bell
from one BDC to the next BDC, as this is a suitable proxy for the interval
between one strike and the next strike.
The energy of the bell is determined by the bell state at the first BDC,
and the inputs are the desired energy at the next BDC, and the desired interval.
Together with force profiles, this uniquely determines the trajectory of the bell
and the amount of checking an pulling required.

Let's say that we need to remove energy A and then remove energy B.  A and B
will be the area under the torque * distance curve on either side of the TDC.
If we just use a 4 segment piecewise linear function of force vs angle, we can
approximate a wide range of pull profiles.  Generally there will be a plateau
of high force, a ramp up to the plateau, a ramp down to zero.  The plateau
may be flat, or may be sloped.

If we assume the effective handstroke may start around 1.5 radians from TDC, and
the effective backstroke may be up to 2 meters for an average height ringer, but
would more naturally be only a little longer than the handstroke (on a large bell).

Because of biomechanics, we want to keep the peak force low, keep the ramp slopes
low, and probably want to reduce the force near TDC, as the force during that
time is not very effective in changing the bell's trajectory.

Muscle fatigue in the hands is determined by force * time, whereas muscle fatigue
and cardio-vascular load are increased by pulling energy, and decreased to some
degree by checking energy that offsets pulling energy on the same stroke.

Perhaps the objective function should use a nonlinear function of "work" and
a nonlinear function of hand and wrist fatigue.  The latter would lead to the
dip in force near TDC and a flattening of the peak forces, and the former
would favor more checking, and distributed energy input over multiple pulls.
A third element should be limits on the rate at which the force can change,
basically limiting it to perhaps 2 to 4 hz bandwidth.

Let's assume a 20-24 cwt bell, with a 1 meter wheel, to keep things simple.
Bigger bells would allow a longer handstroke.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import mplcursors  # type: ignore


class Controller:
    def __init__(
        self,
        loss: float = 0.0,
    ):
        self.loss = loss
        self.moment = 200  # 200 kg-m^2

    def angular_acceleration(self, theta: float, omega: float, omega_dot: float):
        """TODO - create a controller that can achieve a given outcome in interval and energy"""
        return -self.loss * omega


class Bell:

    def __init__(self, controller: Controller, L: float, kappa: float):
        self.controller = controller
        self.L = L
        self.g = 9.81
        omega = math.sqrt(2 * self.g * kappa / L)
        self.state: Tuple[float, float] = 0.0, omega

    def delta_omega_gravity(self, theta: float, omega: float, dt: float) -> float:
        return -self.g / self.L * (np.sin(theta) + np.cos(theta) * dt * omega / 2) * dt

    def delta_omega_gravity_trpz(
        self, theta1: float, theta2: float, dt: float
    ) -> float:
        return -self.g / self.L * (np.sin(theta1) / 2 + np.sin(theta2) / 2) * dt

    def step(self, dt: float) -> Tuple[List[float], float]:
        """Returns the new [theta, omega, angular acceleration], and controller acceleration
        during this interval."""
        # This slightly improves the energy precision and droop,
        # but not enough to drop the second pass
        theta, omega = self.state
        odot_0 = self.delta_omega_gravity(theta, omega, dt) / dt
        odot_1 = (
            odot_0 + self.controller.angular_acceleration(theta, omega, odot_0) / self.L
        )
        odot_2 = (
            odot_0 + self.controller.angular_acceleration(theta, omega, odot_1) / self.L
        )
        o: float = omega + odot_2 * dt
        t: float = theta + (o + omega) * dt / 2
        # Recompute using the initial estimates to do trapezoidal approximation
        # o = omega - loss*dt*(o+omega)/2 - (g/L) * (np.sin(theta)+np.sin(t))/2 * dt
        avg_controller = self.controller.angular_acceleration(
            (theta + t) / 2, (omega + o) / 2, odot_2
        )
        avg_odot = avg_controller + self.delta_omega_gravity_trpz(theta, t, dt) / dt
        o = omega + avg_odot * dt
        t = theta + (o + omega) * dt / 2

        # Add clapper strikes
        self.state = t, o
        # Now compute the angular acceleration at the end state, including the strike impulse.
        odot: float = -self.g / self.L * np.sin(
            t
        ) + self.controller.angular_acceleration(t, o, avg_odot)
        return [t, o, odot], avg_controller


# Define the constants
g = 9.81  # m/s^2
L = 0.8  # m
dt = 0.005
loss = 0  # .01   # Handling loss will require some more work on the kalman filter


# Can we improve on this using:
# sin(a+𝛿) = sin a cos 𝛿 + cos a sin 𝛿
#          ~ sin(a) + 𝛿 cos(a)
def xstep(theta: float, omega: float) -> Tuple[float, float]:
    # This slightly improves the energy precision and droop,
    # but not enough to drop the second pass
    o = (1 - loss * dt) * omega - (g / L) * (
        np.sin(theta) + np.cos(theta) * dt * omega / 2
    ) * dt
    # o = (1-loss*dt) * omega - (g/L) * np.sin(theta) * dt
    t = theta + (o + omega) * dt / 2
    # Recompute using the initial estimates to do trapezoidal approximation
    # o = omega - loss*dt*(o+omega)/2 - (g/L) * (np.sin(theta)+np.sin(t))/2 * dt
    o = omega - loss * dt * (o + omega) / 2 - (g / L) * np.sin(theta / 2 + t / 2) * dt
    t = theta + (o + omega) * dt / 2

    return (t, o)


def main():

    bell = Bell(Controller(), L, 1.99)

    omega_0 = bell.state[1]  # np.sqrt(2 * g * kappa / L)
    theta_0 = 0.0

    # Initialize the lists to store the data
    theta = [theta_0]
    omega = [omega_0]
    odot = [0.0]
    t = [0.0]

    # Iterate through the time steps
    while t[-1] < 40:
        # Update the state
        # (th, o) = xstep(theta[-1], omega[-1])
        ((th, o, odot), cntl) = bell.step(dt)
        omega.append(o)
        theta.append(th)

        t.append(t[-1] + dt)

    theta = np.array(theta)
    omega = np.array(omega)
    print(omega)

    print(theta[0], omega[0])

    # Specs for lsm6dsl:
    # FS = ±2g 80 μg/√Hz  ⇒ 1.2mg @250 Hz bandwidth
    # 4 mdps/√Hz ⇒ 60 mdps ⇒ .001 rps RMS @250 Hz bandwidth
    # So, for the LSM6DSL, we get better performance at 250 Hz, than we get at
    # 100 Hz with the 6050.

    # Plot the energy of the actual motion
    k = 1 - 2 * dt * dt / 3  # Fudge factor
    plt.figure(figsize=(20, 5))
    plt.plot(
        t,
        (1 - np.cos(theta)) + k * L * omega * omega / g / 2,
        label="Raw",
        linewidth=0.5,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title("Large Amplitude Pendulum Energy")

    mplcursors.cursor(hover=True)
    plt.legend()
    plt.show()

    # Plot the results
    plt.figure(figsize=(40, 10))
    plt.plot(t, theta, label="Angle", linewidth=0.5)
    plt.plot(t, omega, label="Angular Velocity", linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (radians)")
    plt.title("Large Amplitude Pendulum")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
