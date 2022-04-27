from torch.optim.optimizer import Optimizer, required
import torch
import math

class AdaJRGS(Optimizer):

    def __init__(self, params,
                 lr=1e-2, radius=1.0, initial_accumulator_value=0,
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
        super(AdaJRGS, self).__init__(params, defaults)

        # Initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['D'] = torch.full_like(
                    p, initial_accumulator_value)
                state['gtil'] = torch.empty_like(p).copy_(p)
                state['x'] = torch.empty_like(p).copy_(p)
                state['sum'] = torch.empty_like(p).copy_(p)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['D'].share_memory_()
                state['gtil'].share_memory_()
                state['x'].share_memory_()
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
                        "AdaJRGS does not support sparse gradients (for now).")

                # x iterate is stored in p.data
                # Receive gradient at x and increase step
                grad = p.grad

                state = self.state[p]
                state['step'] += 1
                t = state['step']

                projected = group['projected']
                radius = group['radius']
                lr = group['lr']

                At = t * (t + 1) / 2
                Att = (t + 1) * (t + 2) / 2
                gamma = 2.0 / lr

                # Compute new eta^2
                newD = torch.add(state['D'],
                    torch.square(torch.add(grad, state['gtil'], alpha=-1)),
                    alpha=t**2)

                # Update s
                dif = torch.add(newD.sqrt(), state['D'].sqrt(), alpha=-1)
                state['sum'].add_(
                    torch.add(
                        torch.mul(dif, state['x']), grad, alpha=-t/gamma)
                )

                # Compute new x
                state['x'] = torch.add(state['sum'], grad, alpha=-(t+1)/gamma)
                # Allow division by 0 since this only happens when
                # elements are already set to 0
                aux = newD.sqrt()
                aux[torch.abs(aux) < 1e-10] = 1.0
                state['x'].div_(aux.add_(group['eps']))

                # Project x
                if projected:
                    state['x'].clamp_(min=-radius, max=radius)

                # Update iterate
                p.data = torch.add(
                    At / Att * p.data,
                    (t + 1) / Att * state['x'])

                state['D'] = newD
                state['gtil'] = grad
        return loss
