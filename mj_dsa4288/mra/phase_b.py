"""
Phase B: put the cryo-EM physics back, one term at a time.

Here SPECTER's forward model is used end-to-end (fixed view, random in-plane
shifts, optional CTF / Poisson noise / detector / ice), so the data no longer
follow the paper's white-Gaussian model. The question is which of the paper's
predictions survive each added term. Noise is set by dose rather than by
``sigma``.

The returned stacks are ordinary ``[n, Y, X]`` tensors. Note two departures
from Phase A:

* shifts are real-space (not cyclic), so ``max_shift`` must keep the particle
  inside the box;
* with a CTF switched on, the estimand is the CTF-filtered projection, not the
  projection itself. Compare against a clean CTF-on reference image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PhysicsConfig:
    """
    Which physics terms to switch on. Defaults reproduce the Phase A limit.

    Attributes
    ----------
    voltage, cs, alpha : float
        Microscope constants (kV, mm, amplitude-contrast ratio).
    defocus_angstrom : float
        Defocus in Angstrom; 0 together with ``cs = 0`` gives no CTF.
    defocus_jitter_angstrom : float
        Half-width of a per-particle uniform defocus perturbation (Phase B.2).
    scattering_model : str
        ``"projection"`` (linear) or ``"multislice"`` (full).
    aberration_model : str
        Kept for old configs; must be ``"holography"`` (the full, non-linear
        transfer function, which SPECTER applies to every wave model).
    noise_model : str or None
        ``None`` (clean) or ``"poisson"`` (dose-limited counting noise).
    detector_model : str or None
        ``None``, ``"perfect"``, ``"k3_300kv"`` ...
    coincidence_radius : float
        Detector coincidence-loss radius in pixels (needs ``noise_model``).
    num_frames : int or None
        Movie frames the dose is fractionated into; coincidence loss is
        applied per frame. None uses SPECTER's default.
    ice_model : str or None
        ``None``, ``"random"`` or ``"gd"``.
    ice_thickness : float
        Ice thickness in Angstrom.
    max_shift_angstrom : float
        Uniform in-plane shift range (real-space, not cyclic).
    bfactor : float or None
        B-factor envelope; deliberately off (Section 3.2 low-pass warning).
    """

    voltage: float = 300.0
    cs: float = 0.0
    alpha: float = 0.0
    defocus_angstrom: float = 0.0
    defocus_jitter_angstrom: float = 0.0
    scattering_model: str = "projection"
    aberration_model: str = "holography"
    noise_model: str | None = None
    detector_model: str | None = None
    coincidence_radius: float = 0.0
    num_frames: int | None = None
    ice_model: str | None = None
    ice_thickness: float = 0.0
    max_shift_angstrom: float = 10.0
    bfactor: float | None = None
    device: str = "cpu"


@dataclass
class ParticleStack:
    """
    Everything SPECTER needs to image one stack; run it with :func:`generate_stack`.

    Attributes
    ----------
    specimen : specter.imaging.Specimen
        The volume, solvated when ice is on.
    poses : specter.imaging.Poses
        Fixed view and the sampled in-plane shift of every particle.
    ctf : specter.imaging.CTF
        Per-particle defocus and Cs.
    exposure : specter.imaging.Exposure
        Dose per particle.
    voltage : float
        kV.
    device : str
        Torch device.
    scatter, optics, coherence, camera : dict
        Keywords for the scattering, transfer-function, coherence and detector
        stages.
    """

    specimen: Any
    poses: Any
    ctf: Any
    exposure: Any
    voltage: float
    device: str
    scatter: dict[str, Any] = field(default_factory=dict)
    optics: dict[str, Any] = field(default_factory=dict)
    coherence: dict[str, Any] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)


def build_generator(
    volume: torch.Tensor,
    pixel_size: float,
    n: int,
    dose: float,
    cfg: PhysicsConfig,
    quaternion: torch.Tensor | None = None,
    seed: int = 0,
) -> ParticleStack:
    """
    Set up a SPECTER particle stack for a fixed view with in-plane shifts.

    Parameters
    ----------
    volume : torch.Tensor
        Scattering potential ``[Z, Y, X]`` (from ``PotentialBuilder``).
    pixel_size : float
        Angstrom per pixel.
    n : int
        Number of particles in the stack.
    dose : float
        Electron dose per square Angstrom (sets the SNR when noise is on).
    cfg : PhysicsConfig
        Which physics to include.
    quaternion : torch.Tensor, optional
        Fixed view; identity if omitted.
    seed : int, optional
        Seed for the shift / defocus sampling.

    Returns
    -------
    ParticleStack
        Pass to :func:`generate_stack` to obtain images ``[n, Y, X]``.
    """
    from specter.imaging import CTF, Exposure, Poses, Specimen

    if cfg.aberration_model != "holography":
        raise ValueError(
            f"aberration_model={cfg.aberration_model!r} is no longer supported; "
            "use 'holography'"
        )
    gen = torch.Generator().manual_seed(seed)
    if quaternion is None:
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
    quats = quaternion[None].float().expand(n, 4).contiguous()
    shifts = (2 * torch.rand(n, 2, generator=gen) - 1) * cfg.max_shift_angstrom
    defocus = torch.full((n,), cfg.defocus_angstrom)
    if cfg.defocus_jitter_angstrom > 0:
        defocus = (
            defocus
            + (2 * torch.rand(n, generator=gen) - 1) * cfg.defocus_jitter_angstrom
        )
    specimen = Specimen(volume.float(), pixel_size).to(cfg.device)
    if cfg.ice_model:
        specimen = specimen.solvate(model=cfg.ice_model, thickness=cfg.ice_thickness)
    return ParticleStack(
        specimen=specimen,
        poses=Poses(quats, shifts, torch.arange(n)),
        ctf=CTF.from_values(defocus, cs=cfg.cs),
        exposure=Exposure(dose=dose, n_frames=cfg.num_frames),
        voltage=cfg.voltage,
        device=cfg.device,
        # SPECTER now oversamples the image grid and applies a chromatic
        # envelope by default; both are switched off to keep the physics above.
        scatter={
            "scattering_model": cfg.scattering_model,
            "alpha": cfg.alpha,
            "oversample": 1,
            "pad_fft": False,
        },
        optics={"bfactor": cfg.bfactor},
        coherence={"cc": None},
        camera={
            "detector": cfg.detector_model,
            "noise": cfg.noise_model,
            "coincidence_radius": cfg.coincidence_radius,
        },
    )


def generate_stack(
    model: ParticleStack, n: int, batch_size: int = 64
) -> torch.Tensor:
    """Image the first ``n`` particles of ``model`` in batches; returns ``[n, Y, X]`` on CPU."""
    from specter.imaging import simulate_particles

    with torch.no_grad():
        images = simulate_particles(
            model.specimen,
            model.poses[:n],
            model.ctf[:n],
            model.exposure,
            voltage=model.voltage,
            batch_size=batch_size,
            scatter=model.scatter,
            optics=model.optics,
            coherence=model.coherence,
            camera=model.camera,
        )
    return images.array.detach().cpu().float()
