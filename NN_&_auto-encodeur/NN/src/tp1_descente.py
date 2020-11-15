import torch
from torch.utils.tensorboard import SummaryWriter
from descente import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    linear_ctx = Context()
    mse_ctx = Context()
    
    y_hat = Linear.forward(x, w, b)

    loss = MSE.forward(y_hat, y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    loss.backward()

    ##  TODO:  Mise à jour des paramètres du modèle
    w.data = w.data - epsilon * w.grad.data
    b.data = b.data - epsilon * b.grad.data

    w.grad.data.zero_()
    b.grad.data.zero_()
    


