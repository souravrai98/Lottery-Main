from torch.optim.optimizer import Optimizer, required
import torch
import math


class AdaACSA(Optimizer):

    def __init__(self, params,
                 lr=1e-2, gamma0=1, radius=None,
                 initial_accumulator_value=1.0, eps=0.0,
                 weight_decay=0.0, beta=1.0, projected=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, gamma0=gamma0, radius=radius,
                        initial_accumulator_value=initial_accumulator_value,
                        eps=eps, weight_decay=weight_decay, beta=beta, projected=projected)
        super(AdaACSA, self).__init__(params, defaults)

        # Initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['gamma'] = group['gamma0']
                state['sum'] = torch.full_like(
                    p, initial_accumulator_value)
                state['x'] = torch.empty_like(p).copy_(p)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()
                state['x'].share_memory_()

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
                        "AdaACSA does not support sparse gradients.")

                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]
                state['step'] += 1

                gamma = state['gamma']
                eta = group['lr']
                projected = group['projected']
                radius = eta if group['radius'] is None else group['radius']

                # If we are solving the unprojected version, update D first
                if not projected:
                    # Slightly decay the preconditioner to avoid
                    # decreasing the step size too much.
                    if group['beta'] < 1.0:
                        state['sum'].mul_(group['beta'])

                    # Update preconditioner before iterating -- improves performance
                    # in the unconstrained case.
                    state['sum'].addcmul_(grad, grad, value=(gamma / eta) ** 2)

                    std = state['sum'].sqrt().add_(group['eps'])
                    state['x'].addcdiv_(grad, std, value=-gamma)


                    new_gamma = (1.0 + math.sqrt(1.0 + 4.0 * gamma * gamma)) * 0.5
                    alpha = gamma * gamma / new_gamma
                    state['gamma'] = new_gamma

                    # New mirror descent iterates are stored as p.data
                    p.data.addcdiv_(grad, std, value=-1).mul_(alpha/(1+alpha))
                    p.data.add_(state['x'], alpha=1 / (1+alpha))
                else:
                    std = state['sum'].sqrt().add_(group['eps'])
                    new_x = torch.addcdiv(state['x'], grad, std, value=-gamma)
                    # For the actual algorithm we would need to clamp 
                    # at eta; instead we clamp at radius to allow more movement
                    new_x.clamp_(min=-radius, max=radius)
                    mov = torch.add(new_x, state['x'], alpha=-1)
                    state['x'] = new_x
                    state['sum'].addcmul_(state['sum'], mov.pow(2), value=(1 / eta) ** 2)

                    new_gamma = (1.0 + math.sqrt(1.0 + 4.0 * gamma * gamma)) * 0.5
                    alpha = gamma * gamma / new_gamma
                    state['gamma'] = new_gamma

                    p.data.add_(mov, alpha=-1/gamma).mul_(alpha / (1 + alpha))
                    p.data.add_(state['x'], alpha=1 / (1 + alpha))
        return loss
