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
from functools import lru_cache
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
        if self.raw_range[0] < 0 or self.raw_range[1] < 0:
            raise ValueError(f"Invalid range: {self.raw_range[0]} {self.raw_range[1]}")
        # if self.raw_range[0] < self.raw_range[1]:
        #     raise ValueError(f"Invalid range: {self.raw_range[0]} {self.raw_range[1]}")

    def shape(self, w_shape: float, f_shape: float) -> Self:
        assert w_shape > 0 and f_shape > 0, f"Invalid shape: {w_shape} {f_shape}"
        assert w_shape >= 0.4, f"Invalid shape: {w_shape} {f_shape}"

        self.w_shape = w_shape
        w_range = (self.raw_range[0] ** w_shape, self.raw_range[1] ** w_shape)
        self.w_center = (w_range[0] + w_range[1]) / 2
        self.w_width = abs(w_range[1] - w_range[0]) / 2
        self.f_shape = f_shape
        return self

    def theta(self, kappa: float, omega: float) -> float:
        """
        Find the angle of the bell at a given angular velocity
        omega = 2 * np.sqrt(9.81 * self.kinetic(theta) / self.L)
        kinetic = self.kappa / 2 - self.potential(theta)
        potential = (1 - np.cos(theta)) / 2
        kinetic = (self.kappa / 2 - (1 - np.cos(theta)) / 2)
        w = 2 * np.sqrt(9.81 * (self.kappa / 2 - (1 - np.cos(theta)) / 2) / self.L)
        (w/2)**2 = 9.81 * (self.kappa / 2 - (1 - np.cos(theta)) / 2) / self.L
        (w/2)**2 * self.L = 9.81 * (self.kappa / 2 - (1 - np.cos(theta)) / 2)
        (w/2)**2 * self.L / 9.81 = (self.kappa - (1 - np.cos(theta)))/2
        2*((w/2)**2 * self.L / 9.81) - self.kappa = - (1 - np.cos(theta))
        1 - np.cos(theta) = self.kappa - 2*((w/2)**2 * self.L / 9.81)
        np.cos(theta) = 1 - self.kappa + 2*((w/2)**2 * self.L / 9.81)
        theta = np.arccos(1 - self.kappa + 2 * ((omega / 2) ** 2 * self.L / 9.81))

        """
        return np.arccos(1 - kappa + 2 * ((omega / 2) ** 2 * self.L / 9.81))

    def transfer(self, k_start: float, k_end: float) -> float:
        """
        This is trivial!
        Find the force required to transfer from k0 to k1 with this profile
        Parameters:
            k0: initial (larger) kappa
            k1: final (smaller) kappa
        """
        assert k_start > k_end, f"Invalid kappa: {k_start} {k_end}"
        a = self.area()
        print(f"Area: {a:.6}")
        f = (k_start - k_end) / a
        sum = 0.0
        step = 0.01  # error is quadratic in step size
        k = k_start
        for w in np.arange(
            min(self.raw_range[0], self.raw_range[1]),
            max(self.raw_range[0], self.raw_range[1]),
            step,
        ):
            ff = f * self.mag(w + step / 2)
            sum += ff * step
            k -= ff * step
            assert k > 0, f"Invalid kappa: {k} {ff} {step} {w}"
        print(
            f"f: {f:.6}, sum: {sum:.6}, k0: {k_start:.6}, k1: {k_end:.6}  error: {k_end-k:.6}"
        )

        return f

    def area(self) -> float:
        """
        The area under the curve of the profile, which estimates total energy transferred
        for an infinitesimal force.
        For each interval, the work is f * dtheta.  dtheta, though is w * dt, and dt
        is 0.1 / w.
        So, we need the sum of f.
        """
        sum = 0.0
        step = 0.01
        for w in np.arange(
            min(self.raw_range[0], self.raw_range[1]),
            max(self.raw_range[0], self.raw_range[1]),
            step,
        ):
            sum += self.mag(w + step / 2)
        return step * sum

    def mag(self, w: float) -> float:
        if self.w_shape != 1.0:
            # BUG - this is sometimes throwing a warning.
            try:
                w = w**self.w_shape
            except RuntimeWarning:
                print(f"Invalid w: {w} {self.w_shape}")
                raise
        q = 1 - ((w - self.w_center) / self.w_width) ** 2
        return math.pow(q, self.f_shape) if q > 0 else 0.0

    def trapezoid(self, w0: float, w1: float) -> float:
        """
        Compute the trapezoidal approximation of the force profile.
        """
        f0 = self.mag(w0)
        f1 = self.mag(w1)
        return (f0 + f1) / 2


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


class Pendulum:
    @staticmethod
    @lru_cache(maxsize=16)
    def omega(L: float, kappa: float, theta: float) -> float:
        return 2 * np.sqrt(9.81 * Pendulum.kinetic(kappa, theta) / L)

    @staticmethod
    @lru_cache(maxsize=16)
    def omega_bdc(L: float, kappa: float) -> float:
        return 2 * np.sqrt(9.81 / L * kappa / 2)

    @staticmethod
    @lru_cache(maxsize=16)
    def theta(L: float, kappa: float, omega: float) -> float:
        cos = 1 - kappa + (omega**2) * L / 9.81 / 2
        if cos > 1:
            return 0.0
        if cos < -1:
            return np.pi
        return np.arccos(cos) if -1 <= cos <= 1 else 0.0

    @staticmethod
    @lru_cache(maxsize=16)
    def kappa(L: float, omega: float, theta: float) -> float:
        return 1 - np.cos(theta) + (omega**2) * L / 9.81 / 2

    @staticmethod
    @lru_cache(maxsize=16)
    def kinetic(kappa: float, theta: float) -> float:
        return kappa / 2 - Pendulum.potential(theta)

    @staticmethod
    @lru_cache(maxsize=16)
    def potential(theta: float) -> float:
        return (1 - np.cos(theta)) / 2

    @staticmethod
    def delta_t(
        L: float, k0: float, w0: float, dk: float, dw: float
    ) -> Tuple[float, float, bool]:
        """Compute the time to go from k0 to w0+dw, and the angle at that point.
        NOTE: Sometimes, kappa > 2, in which case
        when checking, the bell may go over the balance.
        However, the caller should be forcing omega to go
        to zero.  When it does, theta will saturate at pi,
        in the Pendulum.theta() function.  So, the computed
        time will be the time to reach TDC.
        Parameters:
            L: length of pendulum in meters
            k0: initial kappa
            w0: initial omega
            th0: initial theta
            dk: change in kappa
            dw: change in omega
        Returns:
            th1: final theta
            dt: time to go from k0 to k1
            bdc: True if the bell is at BDC
        """
        th0 = Pendulum.theta(L, k0, w0)
        done = False
        assert dw * dk >= 0, f"Invalid dw: {dw} dk: {dk}"
        assert dw != 0, f"Invalid dw: {dw}"

        k1 = k0 + dk
        w1 = w0 + dw
        if dw > 0:
            # When dw is positive, we are pulling the bell.
            # So, omega is increasing, and we need to check that
            # the bell has not reached its BDC (w = w_bdc)
            w_bdc = Pendulum.omega_bdc(L, k1)
            if w1 > w_bdc:
                fraction = (w_bdc - w0) / dw
                w1 = w0 + fraction * dw
                k1 = k0 + fraction * dk
                done = True
        else:
            # When dw is negative, we are checking the bell.
            # So, omega is decreasing, and we need to check that
            # the bell has not reached its peak (w = 0)
            if w1 < 0:
                fraction = -w0 / dw
                w1 = w0 + fraction * dw
                k1 = k0 + fraction * dk
                done = True

        th1 = Pendulum.theta(L, k1, w1)
        w_mid = Pendulum.omega(L, (k0 + k1) / 2, (th0 + th1) / 2)
        # This makes a very tiny difference.
        simpson = (w0 + 4 * w_mid + w1) / 6
        dtheta = th1 - th0
        dt = abs(dtheta / simpson)
        return th1, dt, done


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

    def t0(self) -> float:
        """The natural low amplitude period of the bell"""
        return 2 * np.sqrt(self.L / 9.81) * np.pi

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
        k = self.kinetic(theta)
        assert k >= 0, f"Invalid kinetic: {k} {theta}"
        return 2 * np.sqrt(9.81 * k / self.L)

    def omega_bdc(self) -> float:
        return Pendulum.omega_bdc(self.L, self.kappa)

    def theta(self, omega: float) -> float:
        return Pendulum.theta(self.L, self.kappa, omega)

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

    def pull(
        self,
        kappa: float,
        f: float,
        profile: Profile,
        plot: bool = False,
        step: float = 0.01,
    ) -> Tuple[float, float, NDArray | None]:
        """
        Computes the application of a pulling profile from peak to BDC.
        Pulling (after peak) makes the bell strike earlier.  The delay vs pull
        force is sublinear, close to delta t ~ sqrt(f).
        The
        Returns: the half period, energy, plot data [t, Kappa, PE, Theta, F]

        """

        self.set_height(kappa)

        t = 0.0
        # If kappa > 2.0, this will be pi, and the bell will start at TDC.
        th_start = Pendulum.theta(self.L, self.kappa, 0.0)
        # If kappa > 2.0, this may be non-zero.
        w_start = Pendulum.omega(self.L, self.kappa, th_start)

        # First, find the swing time with no checking, for reference.
        bdc = False
        w0 = w_start
        while not bdc:
            _, dt, bdc = Pendulum.delta_t(self.L, self.kappa, w0, 0.0, step)
            w0 += step
            t += dt
        t_base = t

        time_vals = []
        theta_vals = []
        omega_vals = []
        force_vals = []
        potential_vals = []
        kappa_vals = []

        t = 0.0
        th0 = th_start
        w0 = w_start
        bdc = False
        while not bdc:
            f0 = f * profile.mag(w0)
            time_vals.append(t)
            theta_vals.append(th0)
            omega_vals.append(w0)
            force_vals.append(f0)
            potential_vals.append(self.potential(th0))
            kappa_vals.append(self.kappa)

            f_avg = f * profile.trapezoid(w0, w0 + step)
            th1, dt, bdc = Pendulum.delta_t(self.L, self.kappa, w0, f_avg * step, step)

            self.kappa += f_avg * step
            t += dt
            th0 = th1

            w0 += step
            if bdc:
                break
        f0 = f * profile.mag(w0)
        time_vals.append(t)
        theta_vals.append(th0)
        omega_vals.append(w0)
        force_vals.append(f0)
        potential_vals.append(self.potential(th0))
        kappa_vals.append(self.kappa)

        plot_data: np.ndarray | None = (
            np.array(
                [
                    time_vals,
                    kappa_vals,
                    potential_vals,
                    np.degrees(theta_vals),
                    force_vals,
                    omega_vals,
                ]
            )
            if plot
            else None
        )
        quicker = t_base - t
        print(
            f"Pulling @ {f:.4} raises from {kappa/2:.5} to {self.kappa/2:.5}"
            f" ({(self.kappa-kappa)/2:.6}) and makes strike {quicker:0.3} earlier"
            f" -> {t_base + t:.4}"
        )
        return (t, self.kappa / 2, plot_data)

    def check(
        self, kappa: float, f: float, profile: Profile, plot: bool = False, step=0.01
    ) -> Tuple[float, float, NDArray | None]:
        """
        Computes the application of a checking profile from BDC to peak.
        Returns: the half period, energy, plot data [t, Kappa, PE, Theta, F]
        """

        time_vals = []
        theta_vals = []
        omega_vals = []
        force_vals = []
        potential_vals = []
        kappa_vals = []

        self.set_height(kappa)
        t = 0.0
        th0 = 0.0
        step = -step
        done = False
        w0 = self.omega(th0)
        while not done:
            f0 = f * profile.mag(w0)
            time_vals.append(t)
            theta_vals.append(th0)
            omega_vals.append(w0)
            force_vals.append(f0)
            potential_vals.append(self.potential(th0))
            kappa_vals.append(self.kappa)

            f_avg = f * profile.trapezoid(w0, w0 + step)
            th1, dt, done = Pendulum.delta_t(self.L, self.kappa, w0, f_avg * step, step)
            t += dt
            self.kappa += f_avg * step
            th0 = th1
            w0 += step

        f0 = f * profile.mag(w0)
        time_vals.append(t)
        theta_vals.append(th0)
        omega_vals.append(w0)
        force_vals.append(f0)
        potential_vals.append(self.potential(th0))
        kappa_vals.append(self.kappa)

        plot_data = (
            np.array(
                [
                    time_vals,
                    kappa_vals,
                    potential_vals,
                    np.degrees(theta_vals),
                    force_vals,
                    omega_vals,
                ]
            )
            if plot
            else None
        )
        return (t, self.kappa / 2, plot_data)

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

            # For large forces, this can reduce kappa such that kinetic is negative.
            while True:
                force = (
                    (a - b * (w0 - offset) ** 2) if abs(w0 - offset) < threshold else 0
                )
                self.kappa -= force * dtheta
                if self.kinetic(theta + dtheta) > 0:
                    break
                self.kappa += force * dtheta
                dtheta /= 2

            if first is None and force > 0:
                first = t

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
            assert math.isfinite(dt), f"Invalid dt: {dt} {w0} {w1} {dtheta}"
            dwdt = (w1 - w0) / dt
            if abs(dwdt) < 0.9 * abs(last_dwdt):
                assert False, f"Problem at {t} {theta} {w0} {dwdt} {last_dwdt}"
                break
            last_dwdt = dwdt
            t += dt
            w0 = w1

        np_time = np.array(time_vals)
        if plot:
            np_time = np.array(time_vals)
            # add reversed time to end of time vector
            np_time = np.concatenate([np_time, 2 * np_time[-1] - np_time[::-1]])
            theta_vals = theta_vals + theta_vals[::-1]
            omega_vals = omega_vals + omega_vals[::-1]
            force_vals = force_vals + force_vals[::-1]
            potential_vals = potential_vals + potential_vals[::-1]

        plot_data = np.array(
            [
                np_time,
                np.array(potential_vals),
                np.degrees(theta_vals),
                np.array(force_vals),
            ]
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
        b = 10.0
        while abs(b - a) > 0.00001:
            amp = (a + b) / 2
            period, e_peak, check, theta = self.check_and_pull(kappa, amp, amp / spread)
            # print(f"{a:.5} {amp:.5} {b:.5} => {period:.4} {target:.4} ")
            if period < target:
                b = amp
            else:
                a = amp

        print(f"base = {base:.4}  fraction = {fraction:.4}  target = {target:.4}")
        print(
            f"Found\t{base:.04}\t{kappa/2:.05}\t{amp:.05}\t{period:.04}\t"
            f"{e_peak:.05}\t{e-e_peak:.04}\t{check:.03}"
        )
        assert math.isclose(
            period, target, rel_tol=1e-3
        ), f"Target not matched: {period:.6} != {target:.6}"
        # self.check_and_pull(kappa, amp, amp/spread)
        return (amp, period, e_peak, e - e_peak, check)


def compare_checking(kappa: float, n: int, plot: bool = False) -> None:
    cons = Conserving(0.996, 12000.0, 0.0, 0.0)
    base, e_rounds, _, rounds = cons.check_and_pull(kappa, 0, 1, True)
    p_down = base * (n - 1) / n
    p_up = base * (n + 1) / n
    kappa_base = kappa_for(base)
    kappa_up = kappa_for(p_up)
    _, e_up, _, up = cons.check_and_pull(kappa_up, 0, 1, True)
    kappa_down = kappa_for(p_down)
    print(
        f"n: {n} base: {base:.3}  e_base: {kappa_base / 2:.5} e_down: {kappa_down / 2:.5}"
    )
    print(f"(n-1)/(n+1): {p_down/p_up:.4}")
    impulse_force, period, _, _, _ = cons.find_check(kappa_up, p_down / p_up, 0.2)
    assert math.isclose(
        period, p_down, rel_tol=1e-3
    ), f"Invalid period: {period} {p_down}"

    period, e_impulse, check, impulse = cons.check_and_pull(
        kappa_up, impulse_force, impulse_force / 0.2, True
    )
    width = 7  # 5.3
    # period, e_typical, check, typical = cons.check_and_pull(kappa_b, 0.0867, 0.0867/2.0, True)
    # 6.0	Found	2.448	0.9925	0.04692	2.142	0.9848	0.007671	0.566
    typical_force, _, _, _, _ = cons.find_check(kappa_up, p_down / p_up, width)
    period, e_typical, check, typical = cons.check_and_pull(
        kappa_up, typical_force, typical_force / width, True
    )
    assert math.isclose(
        period, p_down, rel_tol=1e-3
    ), f"Invalid period: {period} {p_down}"

    period, e_down, check, down = cons.check_and_pull(kappa_down, 0, 1, True)
    print(
        f"Intervals: {up[0,-1]:.4}, {rounds[0, -1]:.4}, {impulse[0, -1]:.4}, {typical[0, -1]:.4},"
        f" {down[0, -1]:.4}"
    )
    print(
        f"Energies: {e_up:.5} {e_rounds:.5}  {e_impulse:.5}  {e_typical:.5}  {e_down:.5}"
    )
    print(
        f"Check and pull: {100*(e_up - e_typical):.4}% vs Full Delta: {100*(e_up - e_down):.4}%"
    )
    print(
        f"Check and pull requires {100.0 * (e_up - e_typical) / (e_up - e_down):.4}"
        f"% of energy delta"
    )

    if plot:
        plt.figure(figsize=(12, 6))
        plt.ylim(0.8, 1.0)
        plt.xlim(0.4, 2.3)
        plt.title("Hunting up, down(*), and dodging")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Potential Energy")
        plt.plot(up[0], up[1], label="up", linewidth=0.8)
        plt.plot(rounds[0], rounds[1], label="rounds", linewidth=0.8)
        plt.plot(down[0], down[1], label="down", linewidth=0.8)
        # plt.plot(place[0], place[1], label="rounds", linewidth=0.8)
        plt.plot(impulse[0], impulse[1], label="impulse", linewidth=0.8)
        plt.plot(typical[0], typical[1], label="dodge", linewidth=0.8)
        plt.legend()
        plt.grid(True, "both", "both")
        mplcursors.cursor(hover=True)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.ylim(90, 180)
        plt.ylabel("Theta (degrees)")
        plt.plot(up[0], up[2], label="up", linewidth=0.8)
        plt.plot(rounds[0], rounds[2], label="rounds", linewidth=0.5)
        plt.plot(impulse[0], impulse[2], label="impulse", linewidth=0.5)
        plt.plot(typical[0], typical[2], label="typical", linewidth=0.5)
        plt.plot(down[0], down[2], label="natural", linewidth=0.5)
        plt.legend()
        plt.grid(True, "both", "both")
        mplcursors.cursor(hover=True)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.ylim(0, 180)
        plt.ylabel("Theta (degrees)")
        plt.plot(up[0], up[2], label="up", linewidth=0.5)
        plt.plot(rounds[0], rounds[2], label="rounds", linewidth=0.5)
        plt.plot(impulse[0], impulse[2], label="impulse", linewidth=0.5)
        plt.plot(typical[0], typical[2], label="typical", linewidth=0.5)
        plt.plot(down[0], down[2], label="down", linewidth=0.5)
        plt.plot(typical[0], 1000 * typical[3], label="rope force", linewidth=0.5)
        plt.legend()
        plt.grid(True, "both", "both")
        mplcursors.cursor(hover=True)
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


def test_kappa() -> None:
    L = 0.9969
    o_bdc = Pendulum.omega_bdc(L, 1.99)
    th = Pendulum.theta(L, 1.99, o_bdc * math.sqrt(0.5))
    k = Pendulum.kappa(L, 1.0, th)
    th1 = Pendulum.theta(L, k, 1.0)
    assert th1 == th


def main():
    test_kappa()

    cons = Conserving(0.994, 12000.0, 0.0, 0.0)
    print(f"Natural period: {cons.t0():.6}")
    print(f"Estimated period: {cons.period(1.98):.6}")
    # for step in [0.01, 0.03, 0.1]:
    #     cons.pull(1.98, 0.1, Profile((1.0, 4.0)).shape(0.5, 2.0), True, step)
    # return

    base = 2.5
    target = 7 / 8 * base  # 2.188
    hunting_down_kappa = kappa_for(target * cons.t0() / 2)
    print(f"Hunting down Energy: {hunting_down_kappa/2:.4}")
    # TODO - changes to this profile make a big difference in the delay,
    # and a small difference in the energy.
    prof = Profile((0.1, 4.0)).shape(0.5, 2.0)
    if False:
        rounds_kappa = kappa_for(base)
        amp, p, _, _, _ = cons.find_check(rounds_kappa, 7 / 8, 2.0)
        te = kappa_for(p + 0.09) / 2
        min_te = (
            kappa_for(target + 0.05) / 2
        )  # margin above the natural hunting down height
        print(f"Near check&pull {te:.5}, Minimum {min_te:.5}")

        # Find the period corresponding to the peak energy
        _, p1, _, _, _ = cons.find_check(2 * te, 1.0, 2.0)  # HACK
        print("the change in period is", target / p1)
        amp, p2, e2, _, check = cons.find_check(2 * te, target / p1, 3.0)
        print(f"base: {base:.3}  amp: {amp:.3} for {check:.3} => period: {p:.3} ")

        half, hue, _ = cons.pull(2 * te, 0.014, prof, True)
        print(te, half, hue)

        # 0.3 % margin produces about 50 msec margin that requires checking to correct.
        half, hue, _ = cons.pull(hunting_down_kappa + 2 * 0.003, 0.007, prof, True)

        phd = cons.t0() * t_t0(hunting_down_kappa + 2 * 0.003) / 2
        print(
            f"Hunting down period with margin is: {phd:.6} vs target {target:.6}"
            f"  ... {target/phd:.4}"
        )
        _, phd, _, _, _ = cons.find_check(
            hunting_down_kappa + 2 * 0.003, 1.0, 2.0
        )  # HACK
        amp, p3, e3, _, check = cons.find_check(
            hunting_down_kappa + 2 * 0.003, target / phd, 3.0
        )

        _, phd, _, _, _ = cons.find_check(hunting_down_kappa + 0.003, 1.0, 2.0)  # HACK
        amp, p4, e4, _, check = cons.find_check(
            hunting_down_kappa + 0.004, target / phd, 3.0
        )

        # 0.3 % margin produces about 50 msec margin that requires checking to correct.
        half, hue, _ = cons.pull(hunting_down_kappa + 0.004, 0.003, prof, True)

    print("graded pull forces")
    force = 2 * 0.016
    plt.figure(figsize=(12, 6))
    plt.title("Pulling")
    plt.xlim(180, 90)
    for shape in [0.5, 1.0, 1.5, 2.0, 5.0]:
        prof = Profile((0.2, 4.3)).shape(shape, 0.7)
        cons.set_height(hunting_down_kappa + 0.01)
        plt_th = []
        plt_f = []
        for w in np.linspace(0.0, 6.0, 100):
            plt_th.append(cons.theta(w))
            plt_f.append(prof.mag(w))

        # for force in [0.0, 0.002, 0.004, 0.008, 0.016]:
        t, peak, pull_data = cons.pull(
            hunting_down_kappa, force / prof.area(), prof, True, 0.001
        )
        plt.plot(
            np.degrees(plt_th),
            np.array(plt_f) / prof.area(),
            label=f"shape: {shape:.2} interval: {t:.4}",
        )
        t, end, check_data = cons.check(
            cons.kappa, force / prof.area(), prof, True, step=0.001
        )
        print(
            f"Force: {force:.3}  Energy: {end:.4}  Interval: {1.098 + t:.5}"
            f" (({peak-end:.6}))"
        )
    plt.legend()
    plt.show()
    # with np.printoptions(precision=5, suppress=True):
    #     print(pull_data[5], check_data[5][::-1])
    # for force in [0.0, 0.002, 0.004, 0.008, 0.016]:
    #     cons.pull(1.998, force, prof, True, step=0.001)
    print("\n")
    prof = Profile((0.2, 3.0)).shape(1.5, 1.0)
    for force in [0.0, 0.004, 0.008, 0.016, 0.032]:
        t, peak, pull_data = cons.pull(
            hunting_down_kappa, force / prof.area(), prof, True, 0.01
        )
        t, end, check_data = cons.check(
            cons.kappa, force / prof.area(), prof, True, step=0.01
        )
        print(
            f"Force: {force:.3}  Energy: {end:.4}  Interval: {1.098 + t:.5}"
            f" (({peak-end:.6}))"
        )

    # We end up requiring a force of 0.015 for the check and pull for hunting down, so that
    # we have enough margin to pull the stroke in 2nd place to set up for the point lead.
    _, _, data = cons.check(1.97, 0.01, prof, True)

    if False:
        profiles = [
            Profile((1.0, 4.0)).shape(0.5, 0.8),
            Profile((0.2, 4.0)).shape(0.8, 0.5),
            Profile((1.0, 4.0)).shape(0.9, 0.9),
            Profile((1.0, 4.0)).shape(0.2, 2.0),
            Profile((0.1, 4.0)).shape(0.2, 2.0),
            Profile((0.1, 4.0)).shape(0.2, 1.5),
        ]
        check = True
        for profile in profiles:
            f = profile.transfer(1.97, 1.92)
            if check:
                _, _, data = cons.check(1.97, f, profile, True)
            else:
                _, _, data = cons.pull(1.92, f, profile, True)
            print(
                f"E = [{data[1,0]/2:.4} - {data[1,-1]/2:.4}], Raise = {(data[1,0]-data[1,-1])/2:.4}"
                f"  Period={2*data[0,-1]:.4}  Profile Area:{profile.area():0.4}"
            )
            plt.figure(figsize=(12, 6))
            plt.title("Check or Pull")
            plt.ylim(0, 1)
            plt.xlim(0, data[0][-1])
            if check:
                plt.title("Checking")
            else:
                plt.title("Pulling")
            # plt.ylabel("Theta (degrees)")
            plt.plot(
                data[0], 10 * (1 - data[1] / 2), label="10x delta E", linewidth=0.5
            )
            plt.plot(data[0], data[2], label="PE", linewidth=0.5)
            plt.plot(data[0], 10 * data[4], label="rope force", linewidth=0.5)
            plt.legend()
            mplcursors.cursor(hover=True)
            plt.show()

    hand_and_back()
    compare_checking(1.985, 8, True)


if __name__ == "__main__":
    main()
