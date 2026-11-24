"""Experimental, chunkable exact longitude-FFT group calculation."""
import numpy as np

from lossett.calc.group_increments import _transport
from lossett.filtering.group_kernels import mollifier_fiber_moments


PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def prepare_fields(lat, lon, velocity, grad, gradw, weights, radius):
    nlon = len(lon)
    c, s = np.cos(lon), np.sin(lon)
    rotation = np.zeros((nlon, 3, 3))
    rotation[:, 0, 0] = rotation[:, 1, 1] = c
    rotation[:, 1, 0] = s
    rotation[:, 0, 1] = -s
    rotation[:, 2, 2] = 1.
    u = np.einsum("lpi,ylp->yli", rotation, velocity)
    grad = np.einsum("lpi,ylpq,lqj->ylij", rotation, grad, rotation)
    gradw = np.einsum("lpi,ylp->yli", rotation, gradw)
    radial = np.stack((np.cos(lat), np.zeros(len(lat)), np.sin(lat)), axis=1)
    w = np.sum(u * radial[:, None, :], axis=-1)
    v = u - w[..., None] * radial[:, None, :]
    speed2 = np.sum(u * u, axis=-1)
    projection = np.eye(3) - radial[:, :, None] * radial[:, None, :]
    strain = (grad + np.swapaxes(grad, -1, -2)) / 2 + w[..., None, None] * projection[:, None] / radius
    b = gradw - v / radius
    ev = np.einsum("...ij,...j->...i", strain, v)
    vev = np.sum(v * ev, axis=-1)
    vb = np.sum(v * b, axis=-1)
    div = np.trace(grad, axis1=-2, axis2=-1)
    trace = np.einsum("yij,ylij->yl", projection, strain)
    fields = {
        "one": np.ones_like(w), "speed2": speed2, "w": w, "w2": w * w,
        "w_speed2": w * speed2, "div_speed2": div * speed2, "div": div,
        "div_w": div * w, "vev": vev, "trace": trace, "w_vb": w * vb, "vb": vb,
    }
    for j in range(3):
        fields.update({
            f"v{j}": v[..., j], f"wv{j}": w * v[..., j], f"div_v{j}": div * v[..., j],
            f"ev{j}": ev[..., j], f"b{j}": b[..., j], f"wb{j}": w * b[..., j],
        })
    for j, k in PAIRS:
        fields[f"strain{j}{k}"] = strain[..., j, k]
    spectra = {name: np.fft.rfft(weights * value, axis=1) for name, value in fields.items()}
    return {"radial": radial, "w": w, "v": v, "speed2": speed2, "rotation": rotation,
            "spectra": spectra, "radius": radius}


def build_kernels(lat, nlon, row, scale):
    target_rows = np.flatnonzero(np.abs(lat - lat[row]) <= min(4 * scale, np.pi) + 1e-14)
    target_lat = lat[target_rows]
    delta = 2 * np.pi * np.arange(nlon) / nlon
    c, s = np.cos(delta), np.sin(delta)
    y = np.stack((np.cos(target_lat)[:, None] * c,
                  np.cos(target_lat)[:, None] * s,
                  np.broadcast_to(np.sin(target_lat)[:, None], (len(target_rows), nlon))), axis=-1)
    x = np.array([np.cos(lat[row]), 0., np.sin(lat[row])])
    mu = np.clip(np.sum(y * x, axis=-1), -1, 1)
    g, h, k = mollifier_fiber_moments(mu, scale)
    matrix = np.empty((len(target_rows), nlon, 3, 3))
    for a in range(3):
        transported = _transport(np.eye(3)[a], x, y)
        matrix[..., 0, a] = c * transported[..., 0] + s * transported[..., 1]
        matrix[..., 1, a] = -s * transported[..., 0] + c * transported[..., 1]
        matrix[..., 2, a] = transported[..., 2]
    kernels = [g, k]
    kernels.extend(h * matrix[..., b, a] for a in range(3) for b in range(3))
    for a, aa in PAIRS:
        for b, bb in PAIRS:
            value = matrix[..., b, a] * matrix[..., bb, aa]
            if b != bb:
                value = value + matrix[..., bb, a] * matrix[..., b, aa]
            kernels.append(k * value)
    return target_rows, np.fft.rfft(np.stack(kernels), axis=-1)


def evaluate(fields, row, target_rows, kernels):
    spectra = fields["spectra"]
    nlon = fields["w"].shape[1]

    def correlate(kernel, field):
        return np.fft.irfft(np.sum(np.conj(kernel) * spectra[field][target_rows], axis=0), n=nlon)

    g, k = kernels[:2]
    wx, vx, ux2 = fields["w"][row], fields["v"][row], fields["speed2"][row]
    c0 = correlate(g, "one")
    c2 = correlate(g, "speed2")
    cw = correlate(g, "w")
    cw2 = correlate(g, "w2")
    cw_speed2 = correlate(g, "w_speed2")
    d = np.zeros((nlon, 3))
    dw = np.zeros_like(d)
    div_d = np.zeros_like(d)
    strain_d = np.zeros_like(d)
    b_d = np.zeros_like(d)
    b_w_d = np.zeros_like(d)
    index = 2
    for a in range(3):
        for b in range(3):
            hk = kernels[index]
            index += 1
            for value, name in ((d, f"v{b}"), (dw, f"wv{b}"), (div_d, f"div_v{b}"),
                                (strain_d, f"ev{b}"), (b_d, f"b{b}"), (b_w_d, f"wb{b}")):
                value[:, a] += correlate(hk, name)
    quadratic = np.zeros(nlon)
    for a, aa in PAIRS:
        for b, bb in PAIRS:
            factor = 1 if a == aa else 2
            quadratic += factor * vx[:, a] * vx[:, aa] * correlate(kernels[index], f"strain{b}{bb}")
            index += 1
    dot = np.sum(vx * d, axis=1)
    dotw = np.sum(vx * dw, axis=1)
    structure = c2 + ux2 * c0 - 2 * wx * cw - 2 * dot
    flux = (cw_speed2 - wx * c2 + (ux2 + 2 * wx**2) * cw - wx * ux2 * c0
            - 2 * wx * cw2 - 2 * dotw + 2 * wx * dot)
    pp = np.sum(vx**2, axis=1)
    div_q = (correlate(g, "div_speed2") + ux2 * correlate(g, "div")
             - 2 * wx * correlate(g, "div_w") - 2 * np.sum(vx * div_d, axis=1))
    ae = (correlate(g, "vev") - 2 * np.sum(vx * strain_d, axis=1)
          + quadratic + pp / 2 * correlate(g - k, "trace"))
    bb = (correlate(g, "w_vb") - wx * correlate(g, "vb")
          - np.sum(vx * b_w_d, axis=1) + wx * np.sum(vx * b_d, axis=1))
    horizontal = -0.25 * (div_q + 2 * wx * structure / fields["radius"] + 2 * ae + 2 * bb)
    x = fields["radial"][row]
    mlocal = (dw - wx[:, None] * d - vx * (cw - wx * c0)[:, None]
              + x * (cw2 - 2 * wx * cw + wx**2 * c0)[:, None])
    mixed = np.einsum("lij,lj->li", fields["rotation"], mlocal)
    return np.column_stack((horizontal, structure, flux, mixed, c0))
