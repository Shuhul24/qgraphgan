"""Adversarial objectives: discriminator loss with gradient penalty, and the
REINFORCE generator loss with baseline subtraction and entropy bonus."""

import random

import torch

from .data import distance_two_non_neighbor, random_non_neighbor, train_neighbors
from .models import dot_disc


def emb_gradient_penalty(disc_emb, centers, real_nodes, fake_nodes, gp_lambda):
    """WGAN-GP-style penalty on interpolations between real and fake neighbour embeddings."""
    penalties = []
    for c, r, f in zip(centers, real_nodes, fake_nodes):
        alpha = torch.rand(1, dtype=torch.float64)
        h = (alpha * disc_emb[r] + (1 - alpha) * disc_emb[f]).requires_grad_(True)
        score = torch.sigmoid(torch.dot(disc_emb[c], h))
        grad = torch.autograd.grad(score, h, create_graph=True, retain_graph=True)[0]
        penalties.append((grad.norm() - 1.0).pow(2))
    return gp_lambda * torch.stack(penalties).mean() if penalties else torch.tensor(0.0, dtype=torch.float64)


def disc_loss(disc_emb, gen_emb, projector, gen_wts, centers, generator, train_G, A_train, cfg):
    """Binary cross-entropy on real neighbours vs. mixed negatives.

    Negatives per center: generator samples, one random non-neighbour and one
    distance-2 non-neighbour (first `cfg.k_fake_d` of the pool are used).
    """
    losses, real_nodes, fake_nodes, valid_centers = [], [], [], []
    for c in centers:
        nbrs = train_neighbors(A_train, c)
        if not nbrs:
            continue
        real_scores = torch.stack([dot_disc(c, u, disc_emb) for u in nbrs])
        rl = -torch.log(real_scores + 1e-8).mean()
        fake_loss_terms = []
        fake_pool = []
        with torch.no_grad():
            for _ in range(max(1, cfg.k_fake_d // 2)):
                fn, _ = generator.sample(c, gen_emb.detach(), projector, gen_wts.detach())
                fake_pool.append(fn)
            fake_pool.append(random_non_neighbor(A_train, c))
            fake_pool.append(distance_two_non_neighbor(train_G, A_train, c))
        for fn in fake_pool[:cfg.k_fake_d]:
            fake_loss_terms.append(-torch.log(1.0 - dot_disc(c, fn, disc_emb) + 1e-8))
        losses.append(rl + torch.stack(fake_loss_terms).mean())
        valid_centers.append(c)
        real_nodes.append(random.choice(nbrs))
        fake_nodes.append(fake_pool[0])
    if not losses:
        return torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    gp = emb_gradient_penalty(disc_emb, valid_centers, real_nodes, fake_nodes, cfg.gp_lambda)
    return torch.stack(losses).mean() + gp


def gen_loss(gen_emb, disc_emb, projector, gen_wts, centers, generator, cfg):
    """REINFORCE: reward log D(c, v), clipped to [-reward_clip, 0], with mean-reward baseline.

    An entropy bonus on the exact generator distribution discourages collapse.
    """
    losses, ents = [], []
    for c in centers:
        rewards, logps = [], []
        for _ in range(cfg.n_reinforce):
            sn, lp = generator.sample(c, gen_emb, projector, gen_wts)
            with torch.no_grad():
                r = torch.log(dot_disc(c, sn, disc_emb.detach()) + 1e-8).clamp(-cfg.reward_clip, 0.0)
            rewards.append(r)
            logps.append(lp)
        rt = torch.stack(rewards)
        adv = rt - rt.mean().detach()
        losses.append(torch.stack([-adv[i].detach() * logps[i] for i in range(len(logps))]).mean())
        p = generator.probs(c, gen_emb, projector, gen_wts)
        ents.append(-torch.sum(p * torch.log(p + 1e-8)))
    return torch.stack(losses).mean() - cfg.ent_reg * torch.stack(ents).mean()
