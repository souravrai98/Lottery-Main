from torch.optim.optimizer import Optimizer, required
import torch
import math


class AdaptiveLinearCoupling(Optimizer):

    def __init__(self, params, sigma=0,
                 lr=1e-2, gamma0=1,
                 initial_accumulator_value=1.0, eps=1e-10,
                 weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(sigma=sigma,
                        lr=lr, gamma0=gamma0,
                        initial_accumulator_value=initial_accumulator_value,
                        eps=eps, weight_decay=weight_decay)
        super(AdaptiveLinearCoupling, self).__init__(params, defaults)

        # Initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['gamma'] = group['gamma0'] / group['lr']
                state['sum'] = torch.full_like(
                    p, initial_accumulator_value / group['lr'])
                state['x'] = torch.empty_like(p).copy_(p)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AdaptiveLinearCoupling does not support sparse gradients.")

                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]
                state['step'] += 1

                gamma = state['gamma']
                sigma = group['sigma']
                eta = group['lr']

                state['sum'].addcmul_(grad, grad, value=gamma * gamma)
                # Optional step, turns out not to be required,
                # will keep it for backwards compatibility.
                state['sum'].add_(sigma * sigma, alpha=gamma * gamma)

                std = state['sum'].sqrt()

                # x_new is stored in state['x']
                state['x'].addcdiv_(grad, std, value=-eta * gamma)
                z_plus = torch.addcdiv(p.data, grad, std, value=-1)

                #print('before')
                #print(gamma * eta)
                new_gamma = (1 / eta + math.sqrt(
                    1 / (eta * eta) + 4 * gamma * gamma)) * 0.5
                alpha = eta * gamma * gamma / new_gamma
                state['gamma'] = new_gamma
                #print(new_gamma * eta)

                # New z values are stored as p.data
                p.data = z_plus * (alpha) / (1 + alpha)
                p.data.add_(state['x'], alpha=1 / (1 + alpha))
        return loss
