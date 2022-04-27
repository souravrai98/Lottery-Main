from torch.optim.optimizer import Optimizer, required
import torch
import math

class AdaAGDplus(Optimizer):

    def __init__(self, params,
                 lr=1e-2, radius=1.0, initial_accumulator_value=1,
                 eps=0.0,
                 projected=True):
        if not 0.0 <= radius:
            raise ValueError("Invalid radius value: {}".format(radius))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value))

        defaults = dict(lr=lr, radius=radius,
                        initial_accumulator_value=initial_accumulator_value,
                        eps=eps,
                        projected=projected)
        super(AdaAGDplus, self).__init__(params, defaults)

        # Initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['D'] = torch.full_like(
                    p, initial_accumulator_value)
                state['y'] = torch.empty_like(p).copy_(p)
                state['z'] = p.data.detach().clone()
                state['z0'] = p.data.detach().clone()
                state['sum_grad'] = torch.empty_like(p).copy_(p)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum_grad'].share_memory_()

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
                        "AdaAGD+ does not support sparse gradients (for now).")

                # x iterate is stored in p.data
                # Receive gradient at x and increase step
                grad = p.grad

                state = self.state[p]
                state['step'] += 1
                t = state['step']
                At = t * (t + 1) / 2
                projected = group['projected']
                radius = group['radius']
                lr = group['lr']

                # Update weighted sum of gradients
                state['sum_grad'].add_(grad, alpha=t)

                # Compute gradient step
                std = state['D'].sqrt().add_(group['eps'])
                new_z = torch.addcdiv(
                    state['z0'], state['sum_grad'], std, value=-1)
                # Project onto box
                if projected:
                    new_z.clamp_(min=-radius, max=radius)
                
                sqr_mov = (new_z - state['z']).pow(2)
                state['z'] = new_z

                # Update y
                state['y'] = state['y'] * (At - t) / At + new_z * t / At

                # Update preconditioner
                state['D'].addcmul_(
                    state['D'], sqr_mov,
                    value=1 / lr ** 2)

                # Update x to get gradient info in the next iteration
                p.data = torch.add(
                    At / (At + t + 1) * state['y'],
                    ((t + 1) / (At + t + 1) * state['z']))

        return loss
