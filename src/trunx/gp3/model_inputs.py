"""Class definition for data."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array


class State(NamedTuple):
    """State information."""

    WF: Array  # Foliage mass
    WR: Array  # Root mass
    WS: Array  # Stem mass
    N: Array  # Stems per hectare
    ASW: Array  # Available soil water
    age: Array  # Age in months
    WF_debt: Array  # Foliage biomass stored during dormant period
    prev_month: Array  # Previous month for dormancy transition detection


class ClimateData(NamedTuple):
    """Climate data information."""

    T_avg: jnp.ndarray  # Average monthly temperature (°C)
    T_max: jnp.ndarray  # Maximum monthly temperature (°C)
    VPD: jnp.ndarray  # Vapor pressure deficit (kPa)
    precip: jnp.ndarray  # Monthly precipitation (mm)
    solar_rad: jnp.ndarray  # Monthly solar radiation (MJ/m²)
    frost_days: jnp.ndarray  # Number of frost days in a month
    n_days: jnp.ndarray  # Number of days in each month
    co2: jnp.ndarray  # Atmospheric CO2 concentration (ppm)
    d13catm: jnp.ndarray  # Atmospheric d13C value (‰)
    month: jnp.ndarray  # Current month of climate data
    start_month: np.datetime64  # Start month of climate data


class SiteData(NamedTuple):
    """Site data information."""

    latitude: float  # Latitude of the site
    altitude: float  # Altitude of the site location
    soil_class: int  # Soil parameter for soil class
    ASW: float  # available soil water
    ASW_max: float  # Maximum available soil water
    ASW_min: float  # Minimum available soil water
    year_i: int  # Initial year of simulation
    month_i: int  # Initial month of simulation
    site_start: np.datetime64  # Start month of simulation
    site_end: np.datetime64  # End month of simulation


class SpeciesData(NamedTuple):
    """Species data information."""

    specie: list[str]  # Species name
    FR: jnp.ndarray  # initial site fertility rating for a given species
    WF: jnp.ndarray  # Foliage mass
    WR: jnp.ndarray  # Root mass
    WS: jnp.ndarray  # Stem mass
    N: jnp.ndarray  # Stems per hectare
    planted: list  # Date of planting
    year_p: jnp.ndarray  # Planting year
    month_p: jnp.ndarray  # Planting month


class Params(NamedTuple):
    """Parameter information."""

    # Allocation
    pFS2: jnp.ndarray  # Foliage:stem partitioning ratio @ D = 2 cm
    pFS20: jnp.ndarray  # Foliage:stem partitioning ratio @ D = 20 cm
    aWS: jnp.ndarray  # Constant in the stem mass vs diameter relationship
    nWS: jnp.ndarray  # Power in the stem mass vs diameter relationship
    pRx: jnp.ndarray  # Maximum fraction of NPP allocated to roots
    pRn: jnp.ndarray  # Minimum fraction of NPP allocated to roots

    # Turnover
    gammaF1: jnp.ndarray  # Maximum litterfall rate
    gammaF0: jnp.ndarray  # Litterfall rate at age = 0
    tgammaF: jnp.ndarray  # Age at which litterfall rate has median value
    gammaR: jnp.ndarray  # Average monthly root turnover rate

    # Phenology
    leafgrow: jnp.ndarray  # Month when leaves are produced (deciduous only)
    leaffall: jnp.ndarray  # Month when leaves fall (deciduous only)

    # Temperature modifiers
    Tmin: jnp.ndarray  # Minimum temperature for growth (°C)
    Topt: jnp.ndarray  # Optimum temperature for growth (°C)
    Tmax: jnp.ndarray  # Maximum temperature for growth (°C)
    kF: jnp.ndarray  # Days production lost per frost day

    # Soil water modifiers
    SWconst: (
        jnp.ndarray
    )  # Moisture ratio deficit for fq = 0.5 (c_theta - Landswerg and Waring 1997)
    SWpower: jnp.ndarray  # Power of moisture ratio deficit (n_theta - Landswerg and Waring 1997)

    # CO₂ modifiers
    fCalpha700: jnp.ndarray  # Assimilation enhancement factor at 700 ppm CO₂
    fCg700: jnp.ndarray  # Canopy conductance enhancement factor at 700 ppm CO₂

    # Nutrition
    m0: jnp.ndarray  # Value of m when FR = 0
    fN0: jnp.ndarray  # Value of fNutr when FR = 0
    fNn: jnp.ndarray  # Power of (1 − FR) in fNutr

    # Age modifier
    MaxAge: jnp.ndarray  # Maximum stand age used in age modifier
    nAge: jnp.ndarray  # Power of relative age in fAge function
    rAge: jnp.ndarray  # Relative age giving fAge = 0.5

    # Mortality
    gammaN1: jnp.ndarray  # Mortality rate for large trees
    gammaN0: jnp.ndarray  # Seedling mortality rate (age = 0)
    tgammaN: jnp.ndarray  # Age at which mortality rate has median value
    ngammaN: jnp.ndarray  # Shape of mortality response

    # Self-thinning
    wSx1000: jnp.ndarray  # Max stem mass per tree @ 1000 trees per hectare
    thinPower: jnp.ndarray  # Power in self-thinning rule

    # Biomass loss due to mortality
    mF: jnp.ndarray  # Fraction of foliage biomass lost per dead tree
    mR: jnp.ndarray  # Fraction of root biomass lost per dead tree
    mS: jnp.ndarray  # Fraction of stem biomass lost per dead tree

    # Specific leaf area
    SLA0: jnp.ndarray  # Specific leaf area at age = 0
    SLA1: jnp.ndarray  # Specific leaf area for mature leaves
    tSLA: jnp.ndarray  # Age at which SLA = (SLA0 + SLA1) / 2

    # Light interception
    k: jnp.ndarray  # Extinction coefficient for PAR absorption
    fullCanAge: jnp.ndarray  # Age at canopy closure

    # Rainfall interception
    MaxIntcptn: jnp.ndarray  # Maximum proportion of rainfall intercepted
    LAImaxIntcptn: jnp.ndarray  # LAI for maximum rainfall interception

    # VPD / canopy
    cVPD: jnp.ndarray  # LAI for 50% reduction of VPD in canopy
    alphaCx: jnp.ndarray  # Canopy quantum efficiency

    # Carbon balance
    Y: jnp.ndarray  # Ratio of NPP to GPP

    # Canopy conductance
    MinCond: jnp.ndarray  # Minimum canopy conductance
    MaxCond: jnp.ndarray  # Maximum canopy conductance
    LAIgcx: jnp.ndarray  # LAI for maximum canopy conductance
    CoeffCond: jnp.ndarray  # Defines stomatal response to VPD
    BLcond: jnp.ndarray  # Canopy boundary layer conductance
    RGcGw: jnp.ndarray  # Ratio of CO₂ to H₂O diffusivities in air

    # Carbon isotope discrimination
    D13CTissueDif: jnp.ndarray  # d13C difference between tissue and photosynthate
    aFracDiffu: jnp.ndarray  # Fractionation against 13C in diffusion
    bFracRubi: jnp.ndarray  # Enzymatic fractionation by Rubisco

    # Branch and bark
    fracBB0: jnp.ndarray  # Branch & bark fraction at age = 0
    fracBB1: jnp.ndarray  # Branch & bark fraction for mature stands
    tBB: jnp.ndarray  # Age at which fracBB is at midpoint

    # Wood density
    rhoMin: jnp.ndarray  # Minimum basic wood density (young trees)
    rhoMax: jnp.ndarray  # Maximum basic wood density (older trees)
    tRho: jnp.ndarray  # Age at which rho is at midpoint

    # Crown shape
    crownshape: jnp.ndarray  # Crown shape (1=cone, 2=ellipsoid, 3=half-ellipsoid, 4=rectangular)

    # Height allometry
    aH: jnp.ndarray  # Constant in stem height relationship
    nHB: jnp.ndarray  # Power of DBH in height relationship
    nHC: jnp.ndarray  # Power of competition in height relationship

    # Volume allometry
    aV: jnp.ndarray  # Constant in stem volume relationship
    nVB: jnp.ndarray  # Power of DBH in volume relationship
    nVH: jnp.ndarray  # Power of height in volume relationship
    nVBH: jnp.ndarray  # Power of DBH² × height in volume relationship

    # Crown diameter
    aK: jnp.ndarray  # Constant in crown diameter relationship
    nKB: jnp.ndarray  # Power of DBH in crown diameter relationship
    nKH: jnp.ndarray  # Power of height in crown diameter relationship
    nKC: jnp.ndarray  # Power of competition in crown diameter relationship
    nKrh: jnp.ndarray  # Power of relative height in crown diameter relationship

    # Live crown length
    aHL: jnp.ndarray  # Constant in LCL relationship
    nHLB: jnp.ndarray  # Power of DBH in LCL relationship
    nHLL: jnp.ndarray  # Power of LAI in LCL relationship
    nHLC: jnp.ndarray  # Power of competition in LCL relationship
    nHLrh: jnp.ndarray  # Power of relative height in LCL relationship

    # Radiation balance
    Qa: jnp.ndarray  # Intercept of net vs solar radiation relationship
    Qb: jnp.ndarray  # Slope of net vs solar radiation relationship

    # Constants
    gDM_mol: jnp.ndarray  # Molecular weight of dry matter
    molPAR_MJ: jnp.ndarray  # Conversion of solar radiation to PAR
