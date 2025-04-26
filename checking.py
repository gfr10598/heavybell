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

from typing import Tuple, Self
from numpy.typing import NDArray

import numpy as np
import matplotlib.pyplot as plt
import mplcursors  # type: ignore
from functools import reduce
import math


class Profile:
    def __init__(
        self, w_range: Tuple[float, float], M: float = 12000, L: float = 0.996
    ):
        self.M = M
        self.L = L
        self.raw_range = w_range
        self.w_center = (w_range[0] + w_range[1]) / 2
        self.w_width = abs(w_range[1] - w_range[0]) / 2
        self.w_shape = 1.0
        self.f_shape = 1.0

    def shape(self, w_shape: float, f_shape: float) -> Self:
        self.w_shape = w_shape
        w_range = (self.raw_range[0] ** w_shape, self.raw_range[1] ** w_shape)
        self.w_center = (w_range[0] + w_range[1]) / 2
        self.w_width = abs(w_range[1] - w_range[0]) / 2
        self.f_shape = f_shape
        return self

    def area(self) -> float:
        print(
            "max: ",
            reduce(
                lambda a, b: max(a, self.mag(b)),
                np.arange(self.raw_range[0], self.raw_range[1], 0.01),
            ),
        )
        print("limits: ", self.mag(self.raw_range[0]), self.mag(self.raw_range[1]))
        return 0.01 * reduce(
            lambda a, b: a + b * self.mag(b),
            np.arange(self.raw_range[0], self.raw_range[1], 0.01),
        )

    def mag(self, w: float) -> float:
        if self.w_shape != 1.0:
            w = w**self.w_shape
        q = 1 - ((w - self.w_center) / self.w_width) ** 2
        return math.pow(q, self.f_shape) if q > 0 else 0.0


def t_t0(kappa: float) -> float:
    return 0.90260 - 0.314844 * math.log(1 - kappa / 2)


def kappa_for(t_t0: float) -> float:
    return 2 * (1 - math.exp(-(t_t0 - 0.90260) / 0.314844))


# TODO - make this a class with some shape parameters
def quad(center: float, width: float, x: float):
    # print(f"{x:.3} {1-((x-center)/width)**2}")
    # x = math.sqrt(x)
    q = 1 - (2 * (x - center) / width) ** 2
    return math.pow(q, 0.8) if q > 0 else 0.0


class Conserving:
    def __init__(self, L: float, ft_lbs: float, loss: float, rope: float):
        """
        L: length of pendulum in meters
        ft_lbs: gravitational acceleration in ft/lbs
        loss: energy loss per second
        rope: energy stored in the rope
        """
        self.L = L
        self.ft_lbs = ft_lbs
        self.loss = loss
        self.rope = rope
        self.kappa = 1.9

    def k_rope(self, theta: float) -> float:
        """The approximate energy stored in the rope (K+E)"""
        return 0.0

    def set_height(self, kappa: float) -> None:
        self.kappa = kappa

    def potential(self, theta: float) -> float:
        return (1 - np.cos(theta)) / 2

    def kinetic(self, theta: float) -> float:
        return self.kappa / 2 - self.potential(theta)

    def omega(self, theta: float) -> float:
        return 2 * np.sqrt(9.81 * self.kinetic(theta) / self.L)

    def period(self, kappa: float) -> float:
        """
        TODO - this isn't producing as accurate a result as expected.
        """
        self.set_height(kappa)
        t = 0.0
        theta = 0.0
        dtheta = 0.01
        w0 = self.omega(theta)

        while self.kinetic(theta) > 1e-7:
            w1 = self.omega(theta + dtheta)
            dt = dtheta / ((w0 + w1) / 2)
            t += dt
            w0 = w1
            theta += dtheta
            if dt > 0.001:
                dtheta /= 2.0
            while self.kinetic(theta + dtheta) < 0.8 * self.kinetic(theta):
                dtheta /= 2

        print(t, theta, w0, self.kinetic(0), self.kinetic(theta + dtheta))
        return 2 * t

    def half(
        self, kappa: float, f: float, profile: Profile, plot: bool = False
    ) -> Tuple[float, float, NDArray | None]:
        """
        Returns: the period, energy, plot data
        """

        self.set_height(kappa)
        t = 0.0
        theta = 0.0
        dtheta = 0.1

        w0 = self.omega(theta)
        count = 0

        time_vals = []
        theta_vals = []
        omega_vals = []
        force_vals = []
        potential_vals = []
        kappa_vals = []

        while w0 > 0 and self.kinetic(theta) > 1e-7:
            count += 1
            # if w0 < .1:
            #   print(t, dt, theta, dtheta,  w0)
            while self.kinetic(theta + dtheta) < 0.98 * self.kinetic(theta):
                dtheta /= 2

            force = f * profile.mag(w0)

            self.kappa -= force * dtheta
            self.kappa = max(self.kappa, 0)

            if plot:
                time_vals.append(t)
                theta_vals.append(theta)
                omega_vals.append(w0)
                force_vals.append(force)
                potential_vals.append(self.potential(theta))
                kappa_vals.append(self.kappa)

            theta += dtheta
            w1 = self.omega(theta)
            dt = dtheta / ((w0 + w1) / 2)
            t += dt
            w0 = w1

        plot_data = (
            np.array(
                [
                    time_vals,
                    kappa_vals,
                    potential_vals,
                    np.degrees(theta_vals),
                    force_vals,
                ]
            )
            if plot
            else None
        )
        return (2 * t, self.kappa / 2, plot_data)

    def check_and_pull(
        self, kappa: float, a: float, b: float, plot: bool = False, offset: float = 0.0
    ) -> Tuple[float, float, float, NDArray]:
        """
        Returns: the period, energy, check time
        """
        self.set_height(kappa)
        t = 0.0
        theta = 0.0
        dtheta = 0.1
        w0 = self.omega(theta)
        last_dwdt = 0.0
        threshold = math.sqrt(a / b)
        first = None
        count = 0

        time_vals = []
        theta_vals = []
        omega_vals = []
        force_vals = []
        potential_vals = []
        kappa_vals = []

        while w0 > 0 and self.kinetic(theta) > 1e-7:
            count += 1
            # if w0 < .1:
            #   print(t, dt, theta, dtheta,  w0)
            while self.kinetic(theta + dtheta) < 0.98 * self.kinetic(theta):
                dtheta /= 2

            force = (a - b * (w0 - offset) ** 2) if abs(w0 - offset) < threshold else 0
            if first is None and force > 0:
                first = t

            self.kappa -= force * dtheta
            self.kappa = max(self.kappa, 0)

            if plot:
                time_vals.append(t)
                theta_vals.append(theta)
                omega_vals.append(w0)
                force_vals.append(force)
                potential_vals.append(self.potential(theta))
                kappa_vals.append(self.kappa)

            theta += dtheta
            w1 = self.omega(theta)
            dt = dtheta / ((w0 + w1) / 2)
            dwdt = (w1 - w0) / dt
            if abs(dwdt) < 0.9 * abs(last_dwdt):
                print(f"Stopping at {t} {theta} {w0} {dwdt} {last_dwdt}")
                break
            last_dwdt = dwdt
            t += dt
            w0 = w1

        if plot:
            np_time = np.array(time_vals)
            # add reversed time to end of time vector
            np_time = np_time + (2 * np_time[-1] - np_time[::-1])
            theta_vals = theta_vals + theta_vals[::-1]
            omega_vals = omega_vals + omega_vals[::-1]
            force_vals = force_vals + force_vals[::-1]
            potential_vals = potential_vals + potential_vals[::-1]

        plot_data = np.array(
            [time_vals, potential_vals, np.degrees(theta_vals), force_vals]
        )
        return (2 * t, self.kappa / 2, 0.0 if first is None else t - first, plot_data)

    def find_check(
        self, kappa: float, fraction: float, spread: float = 2
    ) -> tuple[float, float, float, float, float]:
        """
        Returns (amplitude, period, Epeak, deltaE, Tcheck)
        """
        base, e, check, theta = self.check_and_pull(kappa, 0, 1)
        if not kappa / 2 == e:
            raise ValueError(f"kappa/2 != e: {kappa/2} {e}")
        target = base * fraction
        a = 0.0
        b = 5.0
        while abs(b - a) > 0.00001:
            amp = (a + b) / 2
            period, e_peak, check, theta = self.check_and_pull(kappa, amp, amp / spread)
            if period < target:
                b = amp
            else:
                a = amp

        print(
            f"Found\t{base:.04}\t{kappa/2:.05}\t{amp:.05}\t{period:.04}\t"
            f"{e_peak:.05}\t{e-e_peak:.04}\t{check:.03}"
        )
        # self.check_and_pull(kappa, amp, amp/spread)
        return (amp, period, e_peak, e - e_peak, check)


def compare_checking(kappa: float, n: int, plot: bool = False) -> None:
    cons = Conserving(0.996, 12000.0, 0.0, 0.0)
    base, e_unchecked, _, unchecked = cons.check_and_pull(kappa, 0, 1, True)
    kappa_base = kappa_for(base)
    kappa_down = kappa_for(base * (n - 1) / n)
    print(
        f"n: {n} base: {base:.3}  e_base: {kappa_base / 2:.5} e_down: {kappa_down / 2:.5}"
    )
    impulse_force, _, _, _, _ = cons.find_check(kappa_base, (n - 1) / n, 0.1)

    period, e_impulse, check, impulse = cons.check_and_pull(
        kappa_base, impulse_force, impulse_force / 0.1, True
    )
    # period, e_typical, check, typical = cons.check_and_pull(kappa_b, 0.0867, 0.0867/2.0, True)
    # 6.0	Found	2.448	0.9925	0.04692	2.142	0.9848	0.007671	0.566
    typical_force, _, _, _, _ = cons.find_check(kappa_base, 7 / 8, 5.3)
    period, e_typical, check, typical = cons.check_and_pull(
        kappa_base, typical_force, typical_force / 5.32, True
    )  # Spread HACK - WHY?
    period, e_natural, check, natural = cons.check_and_pull(kappa_down, 0, 1, True)
    print(
        f"Intervals: {unchecked[0, -1]:.4}, {impulse[0, -1]:.4}, {typical[0, -1]:.4},"
        f" {natural[0, -1]:.4}"
    )
    print(f"Energies: {e_unchecked:.5}  {e_impulse:.5}  {e_typical:.5}  {e_natural:.5}")
    print(f"{e_unchecked - e_typical:.4} vs {e_unchecked - e_natural:.4}")
    print(f"{(e_unchecked - e_typical) / (e_unchecked - e_natural):.4}")

    if plot:
        plt.figure(figsize=(12, 6))
        plt.ylim(0.9, 1.0)
        plt.xlim(0.7, 1.7)
        plt.ylabel("Potential Energy")
        plt.plot(unchecked[0], unchecked[1], label="unchecked", linewidth=0.5)
        plt.plot(impulse[0], impulse[1], label="impulse", linewidth=0.5)
        plt.plot(typical[0], typical[1], label="typical", linewidth=0.5)
        plt.plot(natural[0], natural[1], label="natural", linewidth=0.5)
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.ylim(90, 180)
        plt.ylabel("Theta (degrees)")
        plt.plot(unchecked[0], unchecked[2], label="unchecked", linewidth=0.5)
        plt.plot(impulse[0], impulse[2], label="impulse", linewidth=0.5)
        plt.plot(typical[0], typical[2], label="typical", linewidth=0.5)
        plt.plot(natural[0], natural[2], label="natural", linewidth=0.5)
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.ylim(0, 180)
        plt.ylabel("Theta (degrees)")
        plt.plot(unchecked[0], unchecked[2], label="unchecked", linewidth=0.5)
        plt.plot(impulse[0], impulse[2], label="impulse", linewidth=0.5)
        plt.plot(typical[0], typical[2], label="typical", linewidth=0.5)
        plt.plot(natural[0], natural[2], label="natural", linewidth=0.5)
        plt.plot(typical[0], 1000 * typical[3], label="rope force", linewidth=0.5)
        plt.legend()
        plt.show()


def natural_vs_checking() -> None:
    cons = Conserving(0.9969, 12000.0, 0.0, 0.0)
    for kappa in [1.985 * 0.999, 1.985 * 1.001]:
        compare_checking(kappa, 8, True)
        # print("\t\t base\t baseE\t Fpeak\t period\t Epeak\t DeltaE\t Tcheck")
        # for spread in np.arange(4.0, 7.0, 0.1):
        #   print(f"{spread:.2}\t", end="")
        #   cons.find_check(1.985, 7/8, spread)
        # return

    # print(cons.period(1.98769), 1.003*t_t0(1.0, 1.98769))
    print("\t base\t baseE\t Fpeak\t period\t Epeak\t DeltaE\t Tcheck")
    for k in np.arange(1.98, 1.9999, 0.001):
        cons.find_check(k, 7 / 8)

    return

    print("\n\n")
    for spread in np.arange(0.1, 4.0, 0.1):
        cons.find_check(1.985, 7 / 8, spread)

    (amp, period, e_peak, delta, check) = cons.find_check(1.985, 7 / 8, 2.0)


def hand_and_back() -> None:
    rope = 0.002  # energy, so delta kappa = 0.004  (1kg*3m / 1000kg*1.5m)
    interval = 0.3  # interval between bell strikes - 3:34 peal speed
    hand = 9 * interval
    back = 8 * interval
    clapper = 0.08  # seconds of offset
    kh = kappa_for(hand - clapper)  # natural kappa through a handstroke
    kb = kappa_for(back + clapper)  # natural kappa through a backstroke
    print(
        f"clapper:  {clapper:.2}  eh-rope: {(kh / 2 - rope) - 1:.5} eb: {kb / 2 - 1:.5}"
        f"  rope: {rope:.3}  whole pull: {hand + back:.3}"
    )

    # Now for hunting down
    # eh-rope: 0.013088 eb: 0.017297  [.015]
    hand = 8 * interval
    back = 7 * interval
    kh = kappa_for(hand - clapper)  # natural kappa through a handstroke
    kb = kappa_for(back + clapper)  # natural kappa through a backstroke
    print(
        f"clapper:  {clapper:.2}  eh-rope: {(kh / 2 - rope) - 1:.5} eb: {kb / 2 - 1:.5}"
        f"  rope: {rope:.3}  whole pull: {hand + back:.3}"
    )


def main():
    cons = Conserving(0.996, 12000.0, 0.0, 0.0)

    profiles = [
        Profile((1.0, 4.0)).shape(0.5, 0.8),
        Profile((1.0, 4.0)).shape(0.8, 0.5),
        Profile((1.0, 4.0)).shape(0.9, 0.9),
        Profile((1.0, 4.0)).shape(0.5, 0.5),
    ]
    for profile in profiles:
        f = 0.04  # 0.2/profile.area()
        _, _, data = cons.half(1.99, f, profile, True)
        print(
            f"E = [{data[1,0]/2:.4} - {data[1,-1]/2:.4}], Raise = {(data[1,0]-data[1,-1])/2:.4}"
            f"  Period={2*data[0,-1]:.4}  Profile Area:{profile.area():0.4}"
        )
        plt.figure(figsize=(12, 6))
        plt.title("foobar")
        plt.ylim(0, 1)
        plt.xlim(data[0][-1], 0)
        plt.ylabel("Theta (degrees)")
        plt.plot(data[0], 10 * (1 - data[1] / 2), label="delta E", linewidth=0.5)
        plt.plot(data[0], data[2], label="height", linewidth=0.5)
        plt.plot(data[0], 10 * data[4], label="rope force", linewidth=0.5)
        plt.legend()
        mplcursors.cursor(hover=True)
        plt.show()

    return
    hand_and_back()
    compare_checking(1.985, 8, True)


if __name__ == "__main__":
    main()
