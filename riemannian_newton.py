import time
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator, minres

from PenCS import ortho_error_val
# Assume the optimizer.py file is in the same directory as this file.
from optimizer import Optimizer, OptimizerResult
from pymanopt.tools.diagnostics import check_gradient, check_hessian
from second_order_landing import T1
from linear_solvers import euc_metric_minres


class RiemannianNewton(Optimizer):
    """
    Riemannian Newton's method optimizer based on line search.

    This optimizer manually implements the core iteration logic of the
    Riemannian Newton's method instead of calling a built-in solver.
    """

    def __init__(self, linesearch_beta=0.5, linesearch_sigma=0.001, theta = 1.0, zeta_max = 0.01, linear_maxiter=200, *args, **kwargs):
        """
        Initializes the optimizer.

        Args:
            linesearch_beta: Reduction factor for the step size in the Armijo line search.
            linesearch_sigma: Constant for the sufficient decrease condition in the Armijo line search.
            *args, **kwargs: Arguments passed to the base Optimizer class.
        """
        super().__init__(*args, **kwargs)
        self._linesearch_beta = linesearch_beta
        self._linesearch_sigma = linesearch_sigma
        self.theta = theta
        self.zeta_max = zeta_max
        self.linear_maxiter = linear_maxiter

    def run(
            self,
            problem,
            *,
            initial_point=None,
            callback: 'Optional[Callable[[np.ndarray, int], Any]]' = None,
            **kwargs,
    ) -> OptimizerResult:
        """
        Run the Riemannian Newton's method on a given optimization problem.

        Args:
            problem: A Pymanopt Problem instance containing the cost, manifold, etc.
            initial_point: The starting point on the manifold.

        Returns:
            The optimization result.
        """
        manifold = problem.manifold
        cost = problem.cost


        # Callback contract (per-iterate):
        #   callback(x, iteration) -> Any
        # The callback is called once per iteration with the current iterate x.
        # If the callback returns True or the string "stop", the optimizer terminates early.
        #
        # Example:
        #   def cb(x, k):
        #       d = subspace_distance(x, X_ref)
        #       history.append(d)
        #       return False
        #
        # Backwards compatibility: callback may also be passed via kwargs.
        if callback is None:
            callback = kwargs.pop("callback", None)


        # 1. Initialization
        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point

        start_time = time.time()
        self._initialize_log(
            optimizer_parameters={"linesearch_beta": self._linesearch_beta, "linesearch_sigma": self._linesearch_sigma})

        if self._verbosity >= 1:
            print(f"Running {self.__class__.__name__}...")

        cost_x = cost(x)

        # Main iteration loop
        for iteration in range(self._max_iterations):
            # 2. Compute the Riemannian gradient and its norm
            egrad_x = problem.euclidean_gradient(x)
            rgrad_x = manifold.projection(x, egrad_x)
            grad_norm = manifold.norm(x, rgrad_x)

            t1 = T1(x, problem.euclidean_gradient)
            t1_norm = manifold.norm(x, t1)

            # Logging and stopping criterion check

            ortho_error = ortho_error_val(x)
            time_elapsed=time.time() - start_time
            if self._log_verbosity >= 1:
                self._add_log_entry(iteration=iteration, point=x, cost=cost_x, tangent_residual=t1_norm,
                                    time=time_elapsed, ortho_error=ortho_error)


            # Per-iteration callback hook (only expose current iterate x)
            if callable(callback):
                try:
                    cb_ret = callback(x, iteration)
                except Exception as ex:
                    raise RuntimeError(f"Callback raised an exception at iteration {iteration}: {ex}") from ex
                if cb_ret is True or cb_ret == "stop":
                    stopping_criterion = "Terminated by callback."
                    break


            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time, iteration=iteration + 1, gradient_norm=grad_norm
            )

            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                break


            # 3. Solve the Newton system: Hess(eta) = -Grad
            # Define the Riemannian Hessian operator
            # Pymanopt's manifold.hess method handles the projection from the
            # Euclidean Hessian to the Riemannian Hessian for us.
            hess_op_callable = lambda eta: manifold.euclidean_to_riemannian_hessian(x, egrad_x, problem.euclidean_hessian(x, eta), eta)

            # Use the Conjugate Gradient (CG) or Minimal Residual (MINRES) method to solve the linear system
            # CG requires vectorized inputs, so we flatten the matrix-form gradient and tangent vectors.

            # b = -rgrad_x
            # b = b.flatten()  # Ensure b is a 1D array
            # def linear_operator_action(eta_flat):
            #     eta = eta_flat.reshape(x.shape)
            #     hess_eta = hess_op_callable(eta)
            #     return hess_eta.flatten()

            # # Create a linear operator for cg to use
            # A = LinearOperator(shape=(x.size, x.size), matvec=linear_operator_action)


            # norm_b = np.linalg.norm(b, ord=2)
            # rtol = min(norm_b ** self.theta, self.zeta_max)
            # eta_flat, info = cg(A, b, rtol=rtol, maxiter=self.linear_maxiter)
            # print("info 状态:", info)
            # eta = eta_flat.reshape(x.shape)  # Reshape the solution back to matrix form


            b = -rgrad_x

            norm_b = np.linalg.norm(b, ord="fro")
            rtol = min(norm_b ** self.theta, self.zeta_max)


            eta, info = euc_metric_minres(hess_op_callable, b, rtol=rtol, maxiter=self.linear_maxiter)


            # ########## test ###########
            # try:
            #     norm_b = np.linalg.norm(b, ord=2)
            #     rtol = min(norm_b ** 1.0, 0.01)
            #     eta, info = minres_matrix(hess_op_callable, -rgrad_x, rtol=rtol)
            #     # print("info 状态:", info)
            #     # eta = eta_flat.reshape(x.shape)  # Reshape the solution back to matrix form
            # ###########################

            # except np.linalg.LinAlgError:
            #     if self._verbosity >= 1:
            #         print("Warning: CG failed to converge, falling back to gradient descent direction.")
            #     eta = -rgrad_x  # Fallback strategy

            # # Check if the Newton direction is a descent direction
            # # If the Hessian is not positive definite, fall back to the gradient descent direction.
            # descent_check = np.real(manifold.inner_product(x, rgrad_x, eta))
            # if descent_check >= 0:
            #     if self._verbosity >= 2:
            #         print(
            #             f"Warning: Newton direction is not a descent direction (descent_check={descent_check:.2e}). Falling back to gradient descent.")
            #     eta = -rgrad_x

            # 4. Armijo backtracking line search
            alpha = 1.0
            # slope = np.real(manifold.inner_product(x, rgrad_x, eta))

            # for _ in range(20):  # Perform a maximum of 20 line search steps
            x_new = manifold.retraction(x, alpha * eta)
            cost_new = cost(x_new)

            #     # Armijo condition: f(x_k + alpha*p_k) <= f(x_k) + c1*alpha*grad(f_k)^T*p_k
            #     if cost_new <= cost_x + self._linesearch_sigma * alpha * slope:
            #         break
            #
            #     alpha *= self._linesearch_beta
            # else:
            #     # If the line search fails, stop the optimization
            #     if self._verbosity >= 1:
            #         print("Line search failed. Could not find a step size satisfying the Armijo condition.")
            #     stopping_criterion = "Terminated - line search failed."
            #     break

            # 5. Update the iterate
            x = x_new
            cost_x = cost_new


            # ** Display:
            if self._verbosity >= 2:
                print(
                    f"k: {iteration:5d}     "
                    f"f: {cost_x:+e}   |grad|: "
                    f"{grad_norm:e}"
                )



            # if self._verbosity >= 2:
            #     print(f"Iter: {iteration + 1:4d}, Cost: {cost_x:.8f}, Grad Norm: {grad_norm:.8e}")



        # Return the final result
        return self._return_result(
            start_time=start_time,
            point=x,
            cost=cost_x,
            iterations=iteration,
            stopping_criterion=stopping_criterion or f"Terminated - max iterations reached after {time.time() - start_time:.2f} seconds.",
            gradient_norm=grad_norm
        )

