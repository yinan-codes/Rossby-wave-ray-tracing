import numpy as np
import matplotlib.pyplot as plt


def validate_first_step(first_step, t0, t_bound):
    """Assert that first_step is valid and return it."""
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")
    if first_step > np.abs(t_bound - t0):
        raise ValueError("`first_step` exceeds bounds.")
    return first_step


def validate_max_step(max_step):
    """Assert that max_Step is valid and return it."""
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")
    return max_step


def validate_tol(rtol, atol, all_dtype='float64'):
    """Validate tolerance values."""
    EPS = np.finfo(all_dtype).eps
    if np.any(rtol < 100 * EPS):
        rtol = np.maximum(rtol, 100 * EPS)
    return rtol, atol


def norm(x, axis=0):
    """Compute RMS norm."""
    return np.linalg.norm(x, axis=axis) / x.shape[0] ** 0.5  # x.size


def select_initial_step(fun, t0, y0, f0, direction, order, rtol, atol):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,samples)
        Initial value of the dependent variable.
    f0 : ndarray, shape (n,samples)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float, (samples,)
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """

    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    h0 = 0.01 * d0 / d1
    h0[d0 < 1e-5] = 1e-6
    h0[d1 < 1e-5] = 1e-6

    # if d0 < 1e-5 or d1 < 1e-5:
    #     h0 = 1e-6
    # else:
    #     h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = norm((f1 - f0) / scale) / h0

    mask1 = np.ones(d1.shape)
    mask2 = np.ones(d2.shape)
    mask1[d1 > 1e-15] = 0
    mask2[d2 > 1e-15] = 0
    h1 = (0.01 / np.nanmax([d1, d2], axis=0)) ** (1 / (order + 1))
    temp = np.maximum(1e-6, h0 * 1e-3)
    h1[mask1 * mask2 == 1] = temp[mask1 * mask2 == 1]

    # if d1 <= 1e-15 and d2 <= 1e-15:
    #     h1 = max(1e-6, h0 * 1e-3)
    # else:
    #     h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return np.min([100 * h0, h1], axis=0)


class OdeSolver:
    """Base class for ODE solvers.

    In order to implement a new solver you need to follow the guidelines:

        1. A constructor must accept parameters presented in the base class
           (listed below) along with any other parameters specific to a solver.
        2. A constructor must accept arbitrary extraneous arguments
           ``**extraneous``, but warn that these arguments are irrelevant
           using `common.warn_extraneous` function. Do not pass these
           arguments to the base class.
        3. A solver must implement a private method `_step_impl(self)` which
           propagates a solver one step further. It must return tuple
           ``(success, message)``, where ``success`` is a boolean indicating
           whether a step was successful, and ``message`` is a string
           containing description of a failure if a step failed or None
           otherwise.
        4. A solver must implement a private method `_dense_output_impl(self)`,
           which returns a `DenseOutput` object covering the last successful
           step.
        5. A solver must have attributes listed below in Attributes section.
           Note that ``t_old`` and ``step_size`` are updated automatically.
        6. Use `fun(self, t, y)` method for the system rhs evaluation, this
           way the number of function evaluations (`nfev`) will be tracked
           automatically.
        7. For convenience, a base class provides `fun_single(self, t, y)` and
           `fun_vectorized(self, t, y)` for evaluating the rhs in
           non-vectorized and vectorized fashions respectively (regardless of
           how `fun` from the constructor is implemented). These calls don't
           increment `nfev`.
        8. If a solver uses a Jacobian matrix and LU decompositions, it should
           track the number of Jacobian evaluations (`njev`) and the number of
           LU decompositions (`nlu`).
        9. By convention, the function evaluations used to compute a finite
           difference approximation of the Jacobian should not be counted in
           `nfev`, thus use `fun_single(self, t, y)` or
           `fun_vectorized(self, t, y)` when computing a finite difference
           approximation of the Jacobian.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
        return an array of the same shape as ``y``. See `vectorized` for more
        information.
    t0 : float
        Initial time. 1-d array, (samples,)
    y0 : array_like, shape (n,samples)
        Initial state.
    t_bound : float
        Boundary time --- the integration won't continue beyond it. It also
        determines the direction of the integration.


    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of the system's rhs evaluations.
    njev : int
        Number of the Jacobian evaluations.
    nlu : int
        Number of LU decompositions.
    """
    TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

    def __init__(self, fun, t0, y0, t_bound):
        self.t_old = None
        self.t = t0 * np.ones((y0.shape[-1]))
        self._fun, self.y = fun, y0
        self.t_bound = t_bound

        fun_single = self._fun

        def fun_vectorized(t, y):
            # f = np.empty_like(y)
            # for i, yi in enumerate(y.T):
            #     f[:, i] = self._fun(t, yi)
            return self.fun_single(t, y)  # f

        def fun(t, y):
            self.nfev += 1
            return self.fun_single(t, y)

        self.fun = fun
        self.fun_single = fun_single
        self.fun_vectorized = fun_vectorized

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.n = self.y.shape[0]  # self.y.size
        self.status = 'running'

        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    @property
    def step_size(self):
        if self.t_old is None:
            return None
        else:
            return np.abs(self.t - self.t_old)

    def step(self):
        """Perform one integration step.

        Returns
        -------
        message : string or None
            Report from the solver. Typically a reason for a failure if
            `self.status` is 'failed' after the step was taken or None
            otherwise.
        """
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished "
                               "solver.")

        if self.n == 0 or (self.t == self.t_bound).all():
            # Handle corner cases of empty solver or no integration.
            self.t_old = self.t
            self.t = self.t_bound
            message = None
            self.status = 'finished'
        else:
            t = self.t
            success, message = self._step_impl()

            if not success:
                self.status = 'failed'
            else:
                self.t_old = t
                if (self.direction * (self.t - self.t_bound) >= 0).all():
                    self.status = 'finished'

        return message

    def _step_impl(self):
        raise NotImplementedError


def rk_step(fun, t, y, f, h, A, B, C, K):
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,samples)
        Current state.
    f : ndarray, shape (n,samples)
        Current value of the derivative, i.e., ``fun(x, y)``.
    h : float
        Step to use.
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        # np.dot(K[:s].T, a[:s]) * h
        # np.einsum('snf,s->nf', K[:s], a[:s])*h
        dy = np.einsum('snf,s->nf', K[:s], a[:s]) * h
        K[s] = fun(t + c * h, y + dy)

    # y + h * np.dot(K[:-1].T, B)

    # np.einsum('snf,s->nf', K[:-1], B)
    y_new = y + h * np.einsum('snf,s->nf', K[:-1], B)
    f_new = fun(t + h, y_new)

    K[-1] = f_new

    return y_new, f_new


class RungeKutta(OdeSolver):
    """Base class for explicit Runge-Kutta methods."""
    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    E: np.ndarray = NotImplemented
    P: np.ndarray = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6,
                 first_step=None,
                 all_dtype='float64',
                 Global_Minstep=0.001):
        # warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound)
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, all_dtype=all_dtype)
        self.f = self.fun(self.t, self.y)
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun, self.t, self.y, self.f, self.direction,
                self.error_estimator_order, self.rtol, self.atol)
        else:
            t_ = t_bound - t0
            first_step[first_step > t_] = t_
            first_step[first_step < 0] = 0
            # temp = select_initial_step(
            #     self.fun, self.t, self.y, self.f, self.direction,
            #     self.error_estimator_order, self.rtol, self.atol)
            self.h_abs = first_step  # np.nanmin([first_step, temp], axis=0)
        self.K = np.empty((self.n_stages + 1, self.n,
                          self.y.shape[-1]), dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None
        self.Global_Minstep = min(Global_Minstep, (t_bound - t0) * 0.001)
        self.SAFETY = 0.9

        self.MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
        self.MAX_FACTOR = 10  # Maximum allowed increase in a step size.

    def _estimate_error(self, K, h):
        # np.einsum('snf,s->nf', K, self.E)  # np.dot(K.T, self.E) * h
        return h[None, :] * np.einsum('snf,s->nf', K, self.E)

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _step_impl(self):
        t = self.t
        y = self.y
        self.f = self.fun(self.t, self.y)
        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step_ = self.Global_Minstep + 0 * \
            np.abs(np.nextafter(t, self.direction * np.inf) - t)
        h_abs_ = np.ones(self.h_abs.shape) - 1 + self.h_abs
        h_abs_[self.h_abs > max_step] = max_step  # max_step is a float
        h_abs_ = np.max([self.h_abs, min_step_], axis=0)
        # print(h_abs_, self.h_abs)

        # if self.h_abs > max_step:
        #     h_abs = max_step
        # elif self.h_abs < min_step:
        #     h_abs = min_step
        # else:
        #     h_abs = self.h_abs

        h_previous = np.ones(h_abs_.shape)
        step_accepted = np.ones(h_abs_.shape) - 1  # False
        step_rejected = np.ones(h_abs_.shape) - 1
        my = np.mean(y, axis=0)
        step_accepted[np.isnan(my)] = 1
        t[np.isnan(my)] = self.t_bound
        step_accepted[t == self.t_bound] = 1
        # print(step_accepted)

        t_new_ = np.ones(t.shape) * t
        y_new_ = np.ones(self.y.shape) * y
        f_new_ = np.ones(self.f.shape) * self.f

        while (step_accepted != 1).any():  # not step_accepted:
            indices = np.where(step_accepted == 0)[0]
            h_abs = h_abs_[indices]
            t_new = t_new_[indices]
            y_new = y_new_[:, indices]
            f_new = f_new_[:, indices]
            # print(h_abs)
            # print(h_abs[h_abs<min_step])
            # = np.nan
            # h_abs_[indices] = h_abs
            # if (h_abs < min_step).all():
            #     print('Too small')
            #     return False, self.TOO_SMALL_STEP
            if np.isnan(h_abs).all():
                print('All nan')
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t[indices] + h

            t_new[self.direction * (t_new - self.t_bound) > 0] = self.t_bound
            # if self.direction * (t_new - self.t_bound) > 0:
            #     t_new = self.t_bound

            h = t_new - t[indices]

            h_abs = np.abs(h)
            # if (h_abs<0.01).any():
            #     print(h_abs)
            temp_k = self.K[..., indices]
            y_new, f_new = rk_step(self.fun, t[indices], y[:, indices], self.f[:, indices], h, self.A,
                                   self.B, self.C, temp_k)
            self.K[..., indices] = temp_k
            scale = atol + \
                np.maximum(np.abs(y[:, indices]), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(
                self.K[..., indices], h, scale)
            error_norm[np.isnan(error_norm)] = 0
            # %%
            # print('h',h_abs,h,'t',t_new,'error',error_norm,'minstep',min_step)
            temp = step_accepted[indices]
            temp[error_norm < 1] = 1
            step_accepted[indices] = temp

            factor = np.minimum(self.MAX_FACTOR, self.SAFETY *
                                error_norm ** self.error_exponent)
            factor[error_norm == 0] = self.MAX_FACTOR

            # temp = step_rejected[indices]
            # for i in range(len(temp)):
            #     # factor[i]>=0; temp[i] = 0 or 1
            #     if temp[i] == 1:
            #         factor[i] = min(temp[i], factor[i])
            factor = np.where(
                # 条件：在 indices 位置上，step_rejected 的值是否为 1
                step_rejected[indices] == 1,
                # 如果条件为 True，计算 min(1, factor[indices])
                np.minimum(1.0, factor),
                # 如果条件为 False，保持 factor[indices] 的原始值
                factor
            )

            h_abs[error_norm < 1] *= factor[error_norm < 1]

            # %%
            factor = np.maximum(self.MIN_FACTOR, self.SAFETY *
                                error_norm ** self.error_exponent)
            h_abs[error_norm >= 1] *= factor[error_norm >= 1]

            temp[error_norm >= 1] = 1
            step_rejected[indices] = temp
            # if error_norm < 1:
            #     if error_norm == 0:
            #         factor = MAX_FACTOR
            #     else:
            #         factor = min(MAX_FACTOR,
            #                      SAFETY * error_norm ** self.error_exponent)

            #     if step_rejected:
            #         factor = min(1, factor)

            #     h_abs *= factor

            #     step_accepted = True
            # else:
            #     h_abs *= max(MIN_FACTOR,
            #                  SAFETY * error_norm ** self.error_exponent)
            #     step_rejected = True

            h_previous[indices] = h
            t_new_[indices] = t_new
            y_new_[:, indices] = y_new
            h_abs_[indices] = h_abs
            f_new_[:, indices] = f_new
        t_new_[np.isnan(t_new_)] = self.t_bound
        # t_new_[np.isnan(my)] = self.t_bound
        # print('all_accepted',t_new_)
        self.h_previous = h_previous
        self.y_old = y

        self.t = t_new_
        self.y = y_new_
        self.h_abs = h_abs_
        self.f = f_new_

        return True, None


class RK45(RungeKutta):
    """Explicit Runge-Kutta method of order 5(4).

    This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method accuracy, but steps are taken
    using the fifth-order accurate formula (local extrapolation is done).
    A quartic interpolation polynomial is used for the dense output [2]_.

    Can be applied in the complex domain.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
        Is always 0 for this solver as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    """
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]
    ])
    B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
    E = np.array([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525,
                  1 / 40])
    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.array([
        [1, -8048581381 / 2820520608, 8663915743 / 2820520608,
         -12715105075 / 11282082432],
        [0, 0, 0, 0],
        [0, 131558114200 / 32700410799, -68118460800 / 10900136933,
         87487479700 / 32700410799],
        [0, -1754552775 / 470086768, 14199869525 / 1410260304,
         -10690763975 / 1880347072],
        [0, 127303824393 / 49829197408, -318862633887 / 49829197408,
         701980252875 / 199316789632],
        [0, -282668133 / 205662961, 2019193451 /
            616988883, -1453857185 / 822651844],
        [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423]])


def rk45_simple_adaptive(fun, t_span, y0, rtol=1e-3,
                         atol=1e-6, all_dtype='float64', first_step=None):
    # t0, t_end = t_span
    t0, t_end = t_span
    y = np.asarray(y0, dtype=all_dtype)  # Ensure y is a float numpy array

    # --- Initialization ---
    # t = t0
    solver = RK45(fun, t0, y, t_end, rtol=rtol,
                  atol=atol, first_step=first_step, all_dtype=all_dtype)
    # --- Main Integration Loop ---
    status = None
    ts = [np.array([t0] * y.shape[-1])]
    ys = [y0]
    # i = 0
    while status is None:
        # print(i)
        # i+=1
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t_old = solver.t_old
        h_abs = solver.h_abs
        t = solver.t
        y = solver.y

        sol = None
        # print(t)
        ts.append(t)
        ys.append(y)
    ts = np.array(ts)
    ys = np.array(ys)
    return ts, ys, h_abs


def rk45_simple_current(fun, t_span, y0, rtol=1e-3, atol=1e-6,
                        all_dtype='float64', first_step=None, t_eval=None, Global_Minstep=0.001):
    # t0, t_end = t_span
    t0, t_end = t_span
    y = np.asarray(y0, dtype=all_dtype)  # Ensure y is a float numpy array

    # --- Initialization ---
    # t = t0
    solver = RK45(fun, t0, y, t_end, rtol=rtol,
                  atol=atol, first_step=first_step, all_dtype=all_dtype, Global_Minstep=Global_Minstep)
    # --- Main Integration Loop ---

    # i = 0
    if t_eval is None:
        t_eval = np.array([t0, t_end])
    else:
        t_eval = np.asarray(t_eval, dtype=all_dtype)

    ts = t_eval
    ys = np.ones((ts.shape[0],) + y0.shape) + np.nan
    ys[0] = y0
    # min_h = np.array([999]*ys.shape[-1])

    for i in range(1, len(t_eval)):
        # if i>100:
        #     solver.Global_Minstep = 100
        solver.t_bound = t_eval[i]
        solver.status = 'running'
        status = None
        while status is None:
            # print(i)
            # i+=1
            message = solver.step()

            if solver.status == 'finished':
                status = 0
            elif solver.status == 'failed':
                status = -1
                break

            # t_old = solver.t_old
            # h_abs = solver.h_abs
            # t = solver.t
            y = solver.y
            # print(t)
        if status != -1:
            ys[i] = y
        else:
            break
    # print(min_h)
    ts = np.array(ts)
    ys = np.array(ys)
    return ts, ys  # , h_abs

# --- Example Usage ---


def simple_rk4(fun, t0, y0, t_bound, dt, all_dtype='float64'):
    '''
    fun: fun(t,y)
    t0, t_bound, dt: float, t_bound>t0
    y0: (vars, samples)
    '''
    dt = float(dt)
    deltat = t_bound - t0
    nt = int(deltat / dt) + 1
    if deltat % dt != 0:
        nt += 1
    nt = int(nt)
    ys = np.ones((nt,) + y0.shape, dtype=all_dtype)
    ts = np.arange(nt) * dt + t0
    if ts[-1] > t_bound:
        ts[-1] = t_bound
    ys[0] = y0
    this_t = t0
    next_t = this_t + dt
    i = 0
    while this_t < t_bound:
        this_dt = dt
        if next_t > t_bound:
            this_dt = dt - next_t + t_bound
            next_t = t_bound

        if this_dt <= 0:
            break
        i += 1
        ts[i] = next_t
        this_y = ys[i - 1]
        k1 = fun(this_t, this_y)
        k2 = fun(this_t + 0.5 * this_dt, this_y + 0.5 * this_dt * k1)
        k3 = fun(this_t + 0.5 * this_dt, this_y + 0.5 * this_dt * k2)
        k4 = fun(this_t + this_dt, this_y + this_dt * k3)
        y_next = this_y + this_dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[i] = y_next
        this_t += dt
        next_t += dt

    return ys[:i + 1], ts[:i + 1]


if __name__ == '__main__':
    # from scipy.integrate import solve_ivp

    def lorenz(t, u, p=10, b=8 / 3, r=28):
        x, y, z = u

        dxdt = -p * x + p * y
        dydt = -x * z + r * x - y
        dzdt = x * y - b * z

        return np.array([dxdt, dydt, dzdt])
    # tstop = 40
    # sol = solve_ivp(
    #     lorenz, (0, tstop), [0.1, 0.1, 0.1],  # <- 追加
    # )
    # t, y, _ = rk45_simple_adaptive(
    #     lorenz, (0, tstop), np.array([[0.1, 0.1, 0.1]]).T,  # <- 追加
    # )

    # t2, y2, _ = rk45_simple_adaptive(
    #     lorenz, (0, tstop), np.array(
    #         [[0.1, 0.1, 0.1], [0.2, 0.3, 0.2]]).T,  # <- 追加
    # )
    # print(np.mean((y[..., 0]-sol.y[:, :].T)**2))
    # t20 = t2[:, 0]
    # stop_ = np.where(t20 >= tstop)[0][0]
    # print(np.mean((y2[0:stop_+1, :, 0]-sol.y[:, :].T)**2))
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # ax.plot(*list(y[:, :, 0].T))
    # plt.show()

    # ys = np.ones((3, 4000))
    # ys[:, 0] = np.array([0.1, 0.1, 0.1])
    # ts = np.linspace(0, 40, 4000)
    # first_step = None
    # for i in range(1, 4000):
    #     solb = solve_ivp(
    #         lorenz, (ts[i-1], ts[i]), ys[:, i-1],
    #     )
    #     ys[:, i] = solb.y[:, -1]

    # ys1 = np.ones((3, 4000))
    # ys1[:, 0] = np.array([0.1, 0.1, 0.1])
    # ts1 = np.linspace(0, 40, 4000)
    # first_step = None
    # for i in range(1, 4000):
    #     t, y, h_abs = rk45_simple_adaptive(
    #         lorenz, (ts1[i-1], ts1[i]), np.array([ys1[:, i-1]]).T,
    #         first_step=first_step
    #     )
    #     first_step = h_abs
    #     ys1[:, i] = y[-1, :, 0]

    ts2 = np.linspace(0, 40, 4001)
    ts2, ys2 = rk45_simple_current(
        lorenz, (0, 40), np.array(
            [[0.1, 0.1, 0.1], [10, 8 / 3, 28]]).T, t_eval=ts2
    )
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(*list(ys2[:, :, 0].T))
    plt.show()

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(*list(ys2[:, :, 1].T))
    plt.show()
    # %%

    def dx(t, u):
        x, = u
        return np.array([2 * t + x * 0])
    ys, ts = simple_rk4(dx, 0, np.array(
        [[0.1], [0.2]]).T, 40, 0.01)
    ts2 = np.linspace(0, 40, 4001)
    ts2, ys2 = rk45_simple_current(
        dx, (0, 40), np.array(
            [[0.1], [0.2]]).T, t_eval=ts2)
    from scipy.integrate import solve_ivp
    ts2 = np.linspace(0, 40, 4001)
    sol = solve_ivp(
        dx, (0, 40), [0.1],  # <- 追加
        t_eval=ts2
    )

    ts2 = np.linspace(0, 40, 4001)
    plt.plot(ys[:, 0, 0] - ts2**2 - 0.1)  # err at 1e-11
    plt.plot(ys2[:, 0, 0] - ts2**2 - 0.1)  # err at 1e-13
    plt.plot(sol.y[0, :] - ts2**2 - 0.1)  # err at 1e-13
    # %%

    def dx(t, u):
        x, = u
        return np.array([np.e**(0.1 * t) + x * 0])
    ys, ts = simple_rk4(dx, 0, np.array(
        [[10], [20]]).T, 40, 0.01)
    ts2 = np.linspace(0, 40, 4001)
    ts2, ys2 = rk45_simple_current(
        dx, (0, 40), np.array(
            [[10], [20]]).T, t_eval=ts2,
        rtol=1e-14,atol=1e-15,
    )
    from scipy.integrate import solve_ivp
    ts2 = np.linspace(0, 40, 4001)
    sol = solve_ivp(
        dx, (0, 40), [10],  # <- 追加
        rtol=1e-14,atol=1e-15,
        t_eval=ts2
    )
    ts2 = np.linspace(0, 40, 4001)
    plt.plot(ys[:, 0, 0] - 10 * np.e**(ts2 * 0.1))  # err at 1e-11
    plt.plot(ys2[:, 0, 0] - 10 * np.e**(ts2 * 0.1))  # err at 1e-13
    plt.plot(sol.y[0, :] - 10 * np.e**(ts2 * 0.1))  # err at 1e-1
