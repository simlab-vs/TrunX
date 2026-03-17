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

    specie: str  # Species name
    FR: float  # initial site fertility rating for a given species
    WF: float  # Foliage mass
    WR: float  # Root mass
    WS: float  # Stem mass
    N: float  # Stems per hectare
    planted: np.datetime64  # Date of planting
    year_p: int  # Planting year
    month_p: int  # Planting month


class Params(NamedTuple):
    """Parameter information."""

    # Allocation
    pFS2: float  # Foliage:stem partitioning ratio @ D = 2 cm
    pFS20: float  # Foliage:stem partitioning ratio @ D = 20 cm
    aWS: float  # Constant in the stem mass vs diameter relationship
    nWS: float  # Power in the stem mass vs diameter relationship
    pRx: float  # Maximum fraction of NPP allocated to roots
    pRn: float  # Minimum fraction of NPP allocated to roots

    # Turnover
    gammaF1: float  # Maximum litterfall rate
    gammaF0: float  # Litterfall rate at age = 0
    tgammaF: float  # Age at which litterfall rate has median value
    gammaR: float  # Average monthly root turnover rate

    # Phenology
    leafgrow: float  # Month when leaves are produced (deciduous only)
    leaffall: float  # Month when leaves fall (deciduous only)

    # Temperature modifiers
    Tmin: float  # Minimum temperature for growth (°C)
    Topt: float  # Optimum temperature for growth (°C)
    Tmax: float  # Maximum temperature for growth (°C)
    kF: float  # Days production lost per frost day

    # Soil water modifiers
    SWconst: float  # Moisture ratio deficit for fq = 0.5 (c_theta - Landswerg and Waring 1997)
    SWpower: float  # Power of moisture ratio deficit (n_theta - Landswerg and Waring 1997)

    # CO₂ modifiers
    fCalpha700: float  # Assimilation enhancement factor at 700 ppm CO₂
    fCg700: float  # Canopy conductance enhancement factor at 700 ppm CO₂

    # Nutrition
    m0: float  # Value of m when FR = 0
    fN0: float  # Value of fNutr when FR = 0
    fNn: float  # Power of (1 − FR) in fNutr

    # Age modifier
    MaxAge: float  # Maximum stand age used in age modifier
    nAge: float  # Power of relative age in fAge function
    rAge: float  # Relative age giving fAge = 0.5

    # Mortality
    gammaN1: float  # Mortality rate for large trees
    gammaN0: float  # Seedling mortality rate (age = 0)
    tgammaN: float  # Age at which mortality rate has median value
    ngammaN: float  # Shape of mortality response

    # Self-thinning
    wSx1000: float  # Max stem mass per tree @ 1000 trees per hectare
    thinPower: float  # Power in self-thinning rule

    # Biomass loss due to mortality
    mF: float  # Fraction of foliage biomass lost per dead tree
    mR: float  # Fraction of root biomass lost per dead tree
    mS: float  # Fraction of stem biomass lost per dead tree

    # Specific leaf area
    SLA0: float  # Specific leaf area at age = 0
    SLA1: float  # Specific leaf area for mature leaves
    tSLA: float  # Age at which SLA = (SLA0 + SLA1) / 2

    # Light interception
    k: float  # Extinction coefficient for PAR absorption
    fullCanAge: float  # Age at canopy closure

    # Rainfall interception
    MaxIntcptn: float  # Maximum proportion of rainfall intercepted
    LAImaxIntcptn: float  # LAI for maximum rainfall interception

    # VPD / canopy
    cVPD: float  # LAI for 50% reduction of VPD in canopy
    alphaCx: float  # Canopy quantum efficiency

    # Carbon balance
    Y: float  # Ratio of NPP to GPP

    # Canopy conductance
    MinCond: float  # Minimum canopy conductance
    MaxCond: float  # Maximum canopy conductance
    LAIgcx: float  # LAI for maximum canopy conductance
    CoeffCond: float  # Defines stomatal response to VPD
    BLcond: float  # Canopy boundary layer conductance
    RGcGw: float  # Ratio of CO₂ to H₂O diffusivities in air

    # Carbon isotope discrimination
    D13CTissueDif: float  # d13C difference between tissue and photosynthate
    aFracDiffu: float  # Fractionation against 13C in diffusion
    bFracRubi: float  # Enzymatic fractionation by Rubisco

    # Branch and bark
    fracBB0: float  # Branch & bark fraction at age = 0
    fracBB1: float  # Branch & bark fraction for mature stands
    tBB: float  # Age at which fracBB is at midpoint

    # Wood density
    rhoMin: float  # Minimum basic wood density (young trees)
    rhoMax: float  # Maximum basic wood density (older trees)
    tRho: float  # Age at which rho is at midpoint

    # Crown shape
    crownshape: float  # Crown shape (1=cone, 2=ellipsoid, 3=half-ellipsoid, 4=rectangular)

    # Height allometry
    aH: float  # Constant in stem height relationship
    nHB: float  # Power of DBH in height relationship
    nHC: float  # Power of competition in height relationship

    # Volume allometry
    aV: float  # Constant in stem volume relationship
    nVB: float  # Power of DBH in volume relationship
    nVH: float  # Power of height in volume relationship
    nVBH: float  # Power of DBH² × height in volume relationship

    # Crown diameter
    aK: float  # Constant in crown diameter relationship
    nKB: float  # Power of DBH in crown diameter relationship
    nKH: float  # Power of height in crown diameter relationship
    nKC: float  # Power of competition in crown diameter relationship
    nKrh: float  # Power of relative height in crown diameter relationship

    # Live crown length
    aHL: float  # Constant in LCL relationship
    nHLB: float  # Power of DBH in LCL relationship
    nHLL: float  # Power of LAI in LCL relationship
    nHLC: float  # Power of competition in LCL relationship
    nHLrh: float  # Power of relative height in LCL relationship

    # Radiation balance
    Qa: float  # Intercept of net vs solar radiation relationship
    Qb: float  # Slope of net vs solar radiation relationship

    # Constants
    gDM_mol: float  # Molecular weight of dry matter
    molPAR_MJ: float  # Conversion of solar radiation to PAR
