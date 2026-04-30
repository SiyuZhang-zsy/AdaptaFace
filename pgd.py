import torch
import torch.nn as nn

from attack import Attack
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

std_tensor = torch.tensor(std, dtype=torch.float32)[None, :, None, None]
mean_tensor = torch.tensor(mean, dtype=torch.float32)[None, :, None, None]

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            mean = mean_tensor.to(adv_images.device)
            std = std_tensor.to(adv_images.device)
            outputs = self.get_logits((adv_images - mean) / std)
            outputs = outputs[2]
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = -loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

# 放在 PGD 类后面
class PGDTargetSource(Attack):
    """
    Scheme B:
    Minimize target CE + lambda_src * source_logit
    = push target up while explicitly pushing source down
    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True, lambda_src=0.25):
        super().__init__("PGDTargetSource", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.lambda_src = lambda_src
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels, source_labels=None):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if source_labels is None:
            raise ValueError("source_labels must be provided for PGDTargetSource.")
        source_labels = source_labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            target_labels = labels

        loss_ce = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            mean = mean_tensor.to(adv_images.device)
            std = std_tensor.to(adv_images.device)
            outputs = self.get_logits((adv_images - mean) / std)
            logits = outputs[2]   # OpenFace AU head

            ce = loss_ce(logits, target_labels)

            src_logit = logits.gather(1, source_labels.unsqueeze(1)).squeeze(1).mean()

            # 我们想最小化： target CE + lambda * source_logit
            # 在当前 attack 框架里，用 cost = -objective 做梯度上升
            objective = ce + self.lambda_src * src_logit
            cost = -objective

            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images
