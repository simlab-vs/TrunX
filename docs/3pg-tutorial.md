# The 3-PG Forest Growth Model: A Tutorial with Differentiable JAX Implementation

**Audience:** Researchers with a quantitative background (ML, CS, applied mathematics) entering process-based ecological modelling.  
**Prerequisites:** Familiarity with Python, JAX basics (`jit`, `grad`, `vmap`, `lax.scan`), and elementary ecology.  
**Goal:** Understand, implement, and calibrate the 3-PG model from first principles.

---

## 1. Why 3-PG?

Forest growth models span a wide spectrum. At one end, **empirical yield tables** fit curves to inventory data — fast, but unable to generalise to novel climates. At the other end, **mechanistic carbon-balance models** (e.g., JULES, LPJ-GUESS, ED2) resolve photosynthesis at the leaf level with hourly time steps and hundreds of parameters — powerful, but expensive and difficult to calibrate.

**3-PG** (Physiological Principles Predicting Growth; Landsberg & Waring, 1997) was designed to occupy the middle ground: a process-based model simple enough to calibrate with standard forestry data, yet physiologically grounded enough to respond credibly to changing climate and management. It operates at the **stand level** (one homogeneous hectare) on a **monthly time step** with roughly 30 species-specific parameters. Since 1997, it has been parameterised for dozens of species on every forested continent (Gupta & Sharma, 2019).

From a machine learning perspective, 3-PG is attractive because its small parameter space and differentiable structure make it a natural candidate for **gradient-based calibration**, **Bayesian inference**, and **hybrid physics-ML architectures** — exactly the kind of scientific model that benefits from modern automatic differentiation frameworks like JAX.

---

## 2. Architecture Overview

The model updates three **biomass pools** — foliage ($W_F$), roots ($W_R$), and stems ($W_S$) — plus **stem count** ($N$) and **available soil water** (ASW) every month. Five coupled submodels interact:

1. **Biomass production:** Sunlight → absorbed radiation → GPP → NPP
2. **Biomass allocation:** NPP split among foliage, roots, and stems
3. **Stem mortality:** Self-thinning removes suppressed trees
4. **Soil water balance:** Rainfall in, transpiration and interception out
5. **Stand properties:** Biomass → DBH, volume, LAI (for forestry outputs)

We implement each submodel as a pure JAX function, then compose them into a single monthly step function wrapped in `jax.lax.scan`.

### Imports and data structures

```python
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from typing import NamedTuple
```

We define the model state as a `NamedTuple`. JAX traces through `NamedTuple` fields cleanly, and the structure makes the code self-documenting:

```python
class State(NamedTuple):
    """Model state at a given time step."""
    WF: float     # Foliage biomass (Mg/ha)
    WR: float     # Root biomass (Mg/ha)
    WS: float     # Stem biomass (Mg/ha)
    N: float      # Stem number (trees/ha)
    ASW: float    # Available soil water (mm)
    age: float    # Stand age (months)
```

Climate inputs arrive as parallel arrays of monthly values:

```python
class ClimateData(NamedTuple):
    T_avg: jnp.ndarray       # Mean monthly temperature (°C)
    VPD: jnp.ndarray         # Mean daytime vapour pressure deficit (kPa)
    precip: jnp.ndarray      # Monthly precipitation (mm)
    solar_rad: jnp.ndarray   # Mean daily solar radiation (MJ/m²/day)
    frost_days: jnp.ndarray  # Frost days per month
    n_days: jnp.ndarray      # Days in each month
```

---

## 3. Submodel 1 — Biomass Production

### 3.1 Biological motivation

All plant growth begins with **photosynthesis**: chloroplasts in leaf cells capture photons and fix atmospheric CO₂ into sugars. At the stand level, the total carbon fixed depends on two things: how much light the canopy intercepts, and how efficiently it converts that light into carbohydrates.

In reality, canopy photosynthesis is fiendishly complex — it depends on leaf angle distributions, sun/shade fractions, nitrogen gradients, mesophyll conductance, Rubisco kinetics, and so on. The genius of 3-PG is to collapse all of this into **two quantities**: absorbed PAR and a single canopy quantum efficiency, modulated by environmental stress factors. This is the **light-use efficiency** (LUE) paradigm introduced by Monteith (1972).

### 3.2 Light interception (Beer's law)

A forest canopy absorbs light following the same exponential attenuation law as a turbid medium. Photosynthetically active radiation (PAR, wavelengths 400–700 nm) is approximately half of total incoming shortwave radiation:

$$\phi_0 = Q \times 0.5 \times n_{\text{days}}$$

The fraction absorbed depends on **leaf area index** (LAI, the total one-sided leaf area per unit ground area, m²/m²) and a **light extinction coefficient** $k$ (typically 0.5 for conifers, 0.6 for broadleaves):

$$\text{APAR} = \phi_0 \,(1 - e^{-k \cdot \text{LAI}})$$

This is Beer's law applied to plant canopies. The extinction coefficient $k$ reflects canopy architecture: horizontally-oriented leaves (broadleaves) intercept more light per unit LAI than vertically-clumped needles.

### 3.3 Environmental growth modifiers

A canopy operating without stress would fix carbon at its maximum quantum efficiency $\alpha_{Cx}$. In practice, several environmental factors reduce this efficiency. Each modifier is a dimensionless function bounded in [0, 1]:

**Temperature modifier $f_T$** — Enzyme kinetics constrain photosynthesis to a species-specific temperature window. Below $T_{\min}$ (typically 0–2 °C for temperate species), metabolic reactions stall. Above $T_{\max}$ (35–45 °C), proteins denature. Peak assimilation occurs at $T_{\text{opt}}$. The bell-shaped response is:

$$f_T = \left(\frac{T - T_{\min}}{T_{\text{opt}} - T_{\min}}\right) \left(\frac{T_{\max} - T}{T_{\max} - T_{\text{opt}}}\right)^{(T_{\max} - T_{\text{opt}})/(T_{\text{opt}} - T_{\min})}$$

```python
def f_temperature(T_avg, T_min, T_opt, T_max):
    """Bell-shaped temperature response of photosynthesis.
    
    Biological basis: enzyme kinetics (Arrhenius + denaturation).
    Returns 1.0 at T_opt, 0.0 at T_min and T_max.
    """
    eps = 1e-8
    a = jnp.clip((T_avg - T_min) / (T_opt - T_min + eps), 0.0, None)
    b = jnp.clip((T_max - T_avg) / (T_max - T_opt + eps), 0.0, None)
    power = (T_max - T_opt) / (T_opt - T_min + eps)
    return jnp.clip(a * (b ** power), 0.0, 1.0)
```

**Frost modifier $f_F$** — Frost damages foliage and disrupts the photosynthetic apparatus. Even frost-tolerant species lose productivity on days when ice crystals form in leaf mesophyll. The modifier linearly penalises frost days:

$$f_F = 1 - k_F \cdot d_F / 30$$

```python
def f_frost(frost_days, k_F):
    """Frost damage reduces monthly photosynthetic capacity.
    
    Biological basis: ice crystal formation damages thylakoid
    membranes and disrupts electron transport chains.
    """
    return jnp.clip(1.0 - k_F * frost_days / 30.0, 0.0, 1.0)
```

**Nutrition modifier $f_N$** — Nitrogen (and to a lesser extent phosphorus) is a key constituent of Rubisco, the enzyme catalysing carbon fixation. Nutrient-poor soils produce less Rubisco per unit leaf area, reducing quantum efficiency. The **fertility rating** FR ∈ [0, 1] is a lumped site quality index:

$$f_N = 1 - (1 - f_{N_0})(1 - \text{FR})$$

```python
def f_nutrition(FR, fN0):
    """Soil fertility effect on canopy photosynthetic capacity.
    
    Biological basis: leaf nitrogen content controls Rubisco
    concentration and therefore carboxylation capacity (Vcmax).
    """
    return 1.0 - (1.0 - fN0) * (1.0 - FR)
```

**VPD modifier $f_D$** — When the air is dry (high vapour pressure deficit), stomata close to prevent desiccation. This simultaneously blocks CO₂ entry into leaves, reducing photosynthesis. The response is exponential, consistent with the observed stomatal sensitivity to VPD across many species (Oren et al., 1999):

$$f_D = e^{-k_D \cdot \text{VPD}}$$

```python
def f_vpd(VPD, k_D):
    """Stomatal closure under atmospheric drought.
    
    Biological basis: guard cells respond to the leaf-to-air
    vapour pressure gradient. High VPD triggers ABA signalling
    and stomatal closure, restricting CO2 diffusion.
    """
    return jnp.exp(-k_D * VPD)
```

**Soil water modifier $f_\theta$** — When the soil dries, root water uptake cannot keep pace with transpirational demand. The plant reduces stomatal conductance to avoid cavitation (air embolism in xylem vessels). A simple linear ramp relative to the maximum soil water holding capacity $\theta_x$:

$$f_\theta = \min\!\left(\frac{\text{ASW}}{c_\theta \cdot \theta_x},\; 1\right)$$

```python
def f_soil_water(ASW, ASW_max, c_theta):
    """Soil drought effect via root-zone water depletion.
    
    Biological basis: as soil matric potential drops, hydraulic
    conductivity decreases, cavitation risk rises, and stomata
    close to maintain xylem integrity.
    """
    return jnp.clip(ASW / (c_theta * ASW_max + 1e-8), 0.0, 1.0)
```

**Age modifier $f_{\text{age}}$** — Old-growth forests are less productive per unit leaf area than young stands. The mechanisms are debated (hydraulic limitation, nutrient immobilisation, increased respiratory costs), but the pattern is robust across biomes. 3-PG captures it with a sigmoidal decline:

$$f_{\text{age}} = \frac{1}{1 + \left(\text{age}/\text{MaxAge}\right)^{n_{\text{age}}} \cdot (1/r_{\text{age}} - 1)}$$

```python
def f_age(age_months, MaxAge, n_age, r_age):
    """Declining productivity with stand age.
    
    Biological basis: hydraulic limitation hypothesis — as trees
    grow taller, the path length and gravitational head increase,
    reducing leaf water potential and forcing stomatal closure
    (Ryan & Yoder, 1997). Also reflects increasing maintenance
    respiration of accumulated sapwood.
    """
    age_years = age_months / 12.0
    rel_age = age_years / (MaxAge + 1e-8)
    return 1.0 / (1.0 + (rel_age ** n_age) * (1.0 / (r_age + 1e-8) - 1.0))
```

### 3.4 GPP and NPP

The modifiers combine multiplicatively. The **physiological modifier** $\varphi$ integrates water- and age-related stress. A critical design choice: 3-PG takes the **minimum** of VPD and soil water stress (rather than multiplying them), because both act through the same pathway — stomatal closure:

$$\varphi = f_{\text{age}} \cdot \min(f_D, f_\theta)$$

Effective quantum efficiency and GPP:

$$\alpha_C = \alpha_{Cx} \cdot f_T \cdot f_F \cdot f_N \cdot \varphi$$
$$\text{GPP} = \alpha_C \cdot \text{APAR}$$

The **NPP/GPP ratio** is the model's boldest simplification. Waring, Landsberg & Williams (1998) showed that across 12 globally distributed forests spanning boreal conifers to subtropical broadleaves, the ratio of net to gross primary production was remarkably stable at **0.47 ± 0.04**. The biological explanation is that construction and maintenance respiration scale roughly proportionally with photosynthesis:

$$\text{NPP} = Y \cdot \text{GPP}, \quad Y \approx 0.47$$

This eliminates the need to model respiration explicitly — a major simplification.

---

## 4. Submodel 2 — Biomass Allocation

### 4.1 Biological motivation

Plants face a fundamental allocation trade-off. Carbon invested in **foliage** increases future light capture but exposes the canopy to wind and herbivory. Carbon invested in **roots** improves water and nutrient uptake but yields no direct photosynthetic return. Carbon invested in **stems** provides mechanical support and competitive height growth.

Ecological theory (the **functional equilibrium** hypothesis; Brouwer, 1963) predicts that plants shift allocation toward the organ that acquires the most limiting resource. A tree on a nutrient-poor, dry site should invest more in roots; one in a dense stand competing for light should invest more in stems.

### 4.2 Root allocation

Root allocation $\eta_R$ increases under stress (low $m$, where $m$ combines nutrition and physiological modifiers):

$$\eta_R = \frac{r_x \cdot r_n}{r_n + (r_x - r_n) \cdot m}$$

When $m \to 0$ (severe stress), $\eta_R \to r_x$ (maximum root allocation, ~0.8). When $m \to 1$ (ideal conditions), $\eta_R \to r_n$ (minimum, ~0.25).

```python
def compute_root_allocation(fN, phi_phys, r_x, r_n):
    """Root allocation increases under resource stress.
    
    Biological basis: functional equilibrium (Brouwer, 1963).
    Plants allocate more C belowground when water or nutrients
    limit growth, and more aboveground when light is limiting.
    """
    m = fN * phi_phys  # combined growth quality index
    return (r_x * r_n) / (r_n + (r_x - r_n) * m + 1e-8)
```

### 4.3 Foliage vs. stem partitioning

The remaining NPP after root allocation is split between foliage and stems via the **foliage-to-stem partitioning ratio** $p_{FS}$, which is an allometric function of mean stem diameter (DBH). Small trees (low DBH) allocate relatively more to foliage to build canopy; large trees (high DBH) allocate more to stems for structural support:

$$p_{FS} = a_p \cdot B^{n_p}$$

The coefficients are derived from two anchor points — the ratio at DBH = 2 cm and DBH = 20 cm:

$$n_p = \frac{\ln(p_{FS20} / p_{FS2})}{\ln(20/2)}, \quad a_p = p_{FS2} \cdot 2^{-n_p}$$

Then:

$$\eta_F = \frac{p_{FS}}{1 + p_{FS}} (1 - \eta_R), \quad \eta_S = \frac{1}{1 + p_{FS}} (1 - \eta_R)$$

```python
def compute_allocation_fractions(B, eta_R, pFS2, pFS20):
    """Split NPP among foliage, roots, and stems.
    
    Biological basis: allometric scaling. Young trees with small
    stems invest heavily in foliage to maximise light capture.
    As trees grow, an increasing fraction goes to structural wood
    needed to support the expanding crown and compete for light.
    """
    np_alloc = jnp.log(pFS20 / (pFS2 + 1e-8) + 1e-8) / jnp.log(10.0)
    ap_alloc = pFS2 * (2.0 ** (-np_alloc))
    pFS = ap_alloc * jnp.clip(B, 0.1, None) ** np_alloc
    
    eta_F = (pFS / (1.0 + pFS)) * (1.0 - eta_R)
    eta_S = (1.0 / (1.0 + pFS)) * (1.0 - eta_R)
    return eta_F, eta_S
```

---

## 5. Submodel 3 — Turnover and Leaf Area Index

### 5.1 Biological motivation

Foliage and fine roots are **ephemeral organs**. Leaves are shed after their photosynthetic returns no longer justify the maintenance costs (nitrogen resorption, defence compounds). Fine roots turn over as soil patches are depleted or as roots are consumed by mycorrhizal fungi and herbivores. This continuous loss is called **litterfall** (foliage) and **root turnover**.

In evergreen conifers, needles may persist 3–8 years; in deciduous broadleaves, the entire canopy is rebuilt annually. 3-PG uses age-dependent monthly turnover rates to capture the transition from low litterfall in young stands (still building canopy) to higher rates in mature stands.

### 5.2 Implementation

```python
def compute_litterfall_rate(age_months, gammaF0, gammaF1, t_gammaF):
    """Age-dependent foliage litterfall rate.
    
    Biological basis: young trees retain foliage longer to build
    LAI rapidly. Mature canopies reach a steady state where 
    litterfall balances new foliage production.
    """
    return gammaF1 + (gammaF0 - gammaF1) * jnp.exp(
        -jnp.log(2.0) * (age_months / (t_gammaF + 1e-8)) ** 2
    )
```

**Leaf area index** couples the foliage pool back to light interception. It is the product of foliage biomass and **specific leaf area** (SLA, m² of leaf per kg of dry mass). SLA declines with age as leaves become thicker and denser:

$$\text{LAI} = W_F \times \text{SLA}(age)$$

```python
def compute_lai(WF, age_months, SLA0, SLA1, t_SLA):
    """LAI from foliage biomass and age-dependent SLA.
    
    Biological basis: young leaves are thin with high area-to-mass
    ratio (high SLA), maximising light capture per unit C invested.
    Mature leaves are thicker with more structural tissue, lower 
    SLA, but greater longevity and stress tolerance.
    """
    SLA = SLA1 + (SLA0 - SLA1) * jnp.exp(
        -jnp.log(2.0) * (age_months / (t_SLA + 1e-8)) ** 2
    )
    return WF * SLA
```

The biomass pools update as:

$$W_F(t+1) = W_F(t) + \eta_F \cdot \text{NPP} - \gamma_F \cdot W_F(t)$$
$$W_R(t+1) = W_R(t) + \eta_R \cdot \text{NPP} - \gamma_R \cdot W_R(t)$$
$$W_S(t+1) = W_S(t) + \eta_S \cdot \text{NPP} - \text{mortality loss}$$

---

## 6. Submodel 4 — Self-Thinning Mortality

### 6.1 Biological motivation

As trees grow, they compete for light, water, and soil resources. In a dense stand, suppressed individuals — those in the lower canopy receiving insufficient light — eventually die. This process is called **self-thinning** and follows one of ecology's few genuine quantitative laws: the **−3/2 power rule** (Yoda et al., 1963; Reineke, 1933).

The law states that in a fully stocked stand, the maximum average tree mass $\bar{w}$ scales with stem density $N$ as:

$$\bar{w}_{\max} = w_{Sx} \cdot (1000/N)^{3/2}$$

where $w_{Sx}$ is the maximum stem mass per tree when there are 1000 stems per hectare (a species-specific constant reflecting maximum tree size).

### 6.2 Differentiable implementation

The original 3-PG uses a hard conditional: if mean tree mass exceeds the self-thinning boundary, kill trees until it no longer does. Hard conditionals break gradient flow. We replace this with a **smooth sigmoid** approximation — when the stand is near the self-thinning boundary, mortality pressure increases continuously:

```python
def apply_self_thinning(WS, N, wSx):
    """Density-dependent mortality via the -3/2 power law.
    
    Biological basis: Yoda's self-thinning rule. As trees grow, 
    canopy space and belowground resources become saturated. The 
    smallest, most suppressed individuals are outcompeted and die, 
    maintaining the stand on a predictable mass-density trajectory.
    
    Implementation note: we use a sigmoid to smooth the step 
    function, preserving differentiability for jax.grad.
    """
    w_s_ind = 1000.0 * WS / (N + 1e-8)                # kg per tree
    w_s_max = wSx * (1000.0 / (N + 1e-8)) ** 1.5      # self-thinning limit
    
    # Smooth mortality pressure: 0 when below limit, rising above it
    mortality_pressure = jax.nn.sigmoid(
        10.0 * (w_s_ind - w_s_max) / (w_s_max + 1e-8)
    )
    mort_frac = jnp.clip(mortality_pressure * 0.05, 0.0, 0.05)
    
    N_new = N * (1.0 - mort_frac)
    # Dead trees are typically smaller than average (suppressed)
    WS_new = WS * (1.0 - mort_frac * 0.8)
    
    return WS_new, jnp.clip(N_new, 1.0, None)
```

---

## 7. Submodel 5 — Soil Water Balance

### 7.1 Biological motivation

Water is often the primary constraint on forest productivity, especially in Mediterranean, subtropical, and seasonally dry climates. The soil acts as a **buffer** between intermittent rainfall and continuous transpirational demand. 3-PG tracks available soil water (ASW) as a simple single-bucket model:

$$\text{ASW}(t+1) = \text{ASW}(t) + P - I - E_T - R_o$$

**Canopy interception** $I$: a fraction of rainfall is caught by leaf surfaces and evaporates before reaching the soil. Dense canopies (high LAI) intercept more:

$$I = i_{\max} \cdot \min(\text{LAI}/L_i,\; 1) \cdot P$$

**Transpiration** $E_T$: water pulled through the soil-plant-atmosphere continuum, driven by the VPD gradient and regulated by stomatal conductance. The full Penman-Monteith equation is complex; 3-PG simplifies it by scaling canopy conductance $G_c$ with LAI and the physiological modifier:

$$G_c = g_{Cx} \cdot \min(\text{LAI}/L_g,\; 1) \cdot \varphi$$

**Runoff** $R_o$: any water exceeding the soil's holding capacity ($\theta_x$) is lost as drainage.

```python
def compute_water_balance(ASW, precip, LAI, phi_phys, VPD, n_days,
                          ASW_max, g_cx, LAI_gc, i_max, LAI_interc):
    """Monthly soil water balance.
    
    Biological basis: the soil-plant-atmosphere continuum. Trees 
    are hydraulic conduits: water moves from soil (high matric 
    potential) through roots, xylem, and stomata to the atmosphere 
    (low water potential). The flux rate depends on the VPD driving 
    gradient and on stomatal/canopy conductance.
    """
    # Canopy interception (rain caught by leaves, re-evaporated)
    interception = i_max * jnp.minimum(LAI / (LAI_interc + 1e-8), 1.0) * precip
    
    # Canopy conductance (m/s), reduced by water/VPD stress
    Gc = g_cx * jnp.minimum(LAI / (LAI_gc + 1e-8), 1.0) * phi_phys
    
    # Simplified transpiration (mm/month)
    # The factor 20 approximates the Penman-Monteith conversion
    # (m/s * kPa * days → mm/month) for typical conditions
    ET = Gc * VPD * n_days * 20.0
    
    ASW_new = ASW + precip - interception - ET
    return jnp.clip(ASW_new, 0.0, ASW_max)
```

---

## 8. Stand-Level Outputs

### 8.1 DBH and volume

Forest managers care about **diameter at breast height** (DBH, measured at 1.3 m) and **stand volume** (m³/ha). These are derived from stem biomass via allometric relationships. The allometry $w_s = a_S \cdot B^{n_S}$ (where $w_s$ is individual stem mass in kg and $B$ is DBH in cm) encodes species-specific wood density and stem taper. Inverted:

$$B = \left(\frac{1000 \cdot W_S}{a_S \cdot N}\right)^{1/n_S}$$

Volume is obtained by correcting stem biomass for bark and branches ($f_{BB}$, typically ~15%) and dividing by basic wood density $\rho$:

$$V = W_S \cdot (1 - f_{BB}) / \rho$$

```python
def compute_dbh(WS, N, aS, nS):
    """Mean DBH from stand-level stem biomass.
    
    Biological basis: allometric scaling laws. The relationship
    between tree diameter and biomass follows power laws that 
    emerge from vascular network geometry and mechanical support 
    requirements (West, Brown & Enquist, 1999).
    """
    w_s_ind = 1000.0 * WS / (N + 1e-8)  # kg per tree
    return jnp.clip(w_s_ind / (aS + 1e-8), 1e-8, None) ** (1.0 / nS)
```

---

## 9. Assembling the Monthly Step

All submodels compose into a single function that advances the state by one month:

```python
def model_step(state, climate_month, params, site):
    """One monthly time step of 3-PG.
    
    Computes: LAI → light interception → modifiers → GPP → NPP
    → allocation → biomass update → mortality → water balance.
    """
    T_avg, VPD, precip, solar_rad, frost_days, n_days = climate_month
    WF, WR, WS, N, ASW, age_months = state

    # --- Leaf area ---
    LAI = compute_lai(WF, age_months, params.SLA0, params.SLA1, params.t_SLA)
    LAI = jnp.clip(LAI, 0.0, 15.0)

    # --- Light interception (Beer's law) ---
    phi0 = solar_rad * 0.5 * n_days
    APAR = phi0 * (1.0 - jnp.exp(-params.k * LAI))

    # --- Growth modifiers ---
    fT   = f_temperature(T_avg, params.T_min, params.T_opt, params.T_max)
    fF   = f_frost(frost_days, params.k_F)
    fN   = f_nutrition(site.FR, params.fN0)
    fD   = f_vpd(VPD, params.k_D)
    fSW  = f_soil_water(ASW, site.ASW_max, params.c_theta)
    fAge = f_age(age_months, params.MaxAge, params.n_age, params.r_age)

    phi_phys = fAge * jnp.minimum(fD, fSW)
    alpha_c  = params.alpha_cx * fT * fF * fN * phi_phys

    # --- Production ---
    GPP = alpha_c * APAR
    NPP = params.Y * GPP

    # --- Allocation ---
    eta_R = compute_root_allocation(fN, phi_phys, params.r_x, params.r_n)
    B = compute_dbh(WS, N, params.aS, params.nS)
    eta_F, eta_S = compute_allocation_fractions(B, eta_R, params.pFS2, params.pFS20)

    # --- Turnover ---
    gammaF = compute_litterfall_rate(
        age_months, params.gammaF0, params.gammaF1, params.t_gammaF
    )

    # --- Biomass update ---
    WF_new = jnp.clip(WF + eta_F * NPP - gammaF * WF, 0.01, None)
    WR_new = jnp.clip(WR + eta_R * NPP - params.gammaR * WR, 0.01, None)
    WS_new = jnp.clip(WS + eta_S * NPP, 0.01, None)

    # --- Self-thinning ---
    WS_new, N_new = apply_self_thinning(WS_new, N, params.wSx)

    # --- Water balance ---
    ASW_new = compute_water_balance(
        ASW, precip, LAI, phi_phys, VPD, n_days,
        site.ASW_max, params.g_cx, params.LAI_gc,
        params.i_max, params.LAI_interc,
    )

    new_state = State(WF=WF_new, WR=WR_new, WS=WS_new,
                      N=N_new, ASW=ASW_new, age=age_months + 1.0)

    outputs = dict(GPP=GPP, NPP=NPP, LAI=LAI, DBH=B,
                   fT=fT, fD=fD, fSW=fSW, fAge=fAge,
                   WF=WF_new, WR=WR_new, WS=WS_new, N=N_new,
                   Volume=WS_new * 0.85 / (params.rho + 1e-8))

    return new_state, outputs
```

---

## 10. Simulation Loop with `jax.lax.scan`

`lax.scan` is JAX's functional loop primitive. It is essential for JIT compilation — a Python `for` loop would unroll the computational graph across 360 months (30 years), producing enormous trace times and memory usage. `lax.scan` compiles to a fixed-size loop:

```python
def run_3pg(initial_state, climate, params, site):
    """Full 3-PG simulation via jax.lax.scan.
    
    Returns the final state and a dict of monthly output arrays.
    """
    climate_stack = jnp.stack([
        climate.T_avg, climate.VPD, climate.precip,
        climate.solar_rad, climate.frost_days, climate.n_days,
    ], axis=-1)  # shape: [n_months, 6]

    def scan_fn(state, climate_arr):
        cm = (climate_arr[0], climate_arr[1], climate_arr[2],
              climate_arr[3], climate_arr[4], climate_arr[5])
        return model_step(state, cm, params, site)

    final_state, all_outputs = jax.lax.scan(scan_fn, initial_state, climate_stack)
    return final_state, all_outputs
```

---

## 11. Exploiting Differentiability

### 11.1 Sensitivity analysis via `jax.grad`

Because every function is composed of smooth JAX primitives (with sigmoid replacements for hard conditionals), we can differentiate the **entire 30-year simulation** with respect to any parameter:

```python
def ws_final(alpha_cx, k_D, Y_val, params, initial_state, climate, site):
    """Scalar function: final stem biomass as a function of 3 parameters."""
    p = params._replace(alpha_cx=alpha_cx, k_D=k_D, Y=Y_val)
    final, _ = run_3pg(initial_state, climate, p, site)
    return final.WS

# Gradients of 30-year stem biomass w.r.t. key parameters
grad_alpha = grad(ws_final, argnums=0)(0.045, 1.5, 0.47, params, s0, clim, site)
grad_kD    = grad(ws_final, argnums=1)(0.045, 1.5, 0.47, params, s0, clim, site)
grad_Y     = grad(ws_final, argnums=2)(0.045, 1.5, 0.47, params, s0, clim, site)
```

This replaces finite-difference sensitivity analysis (which requires $2p$ model evaluations for $p$ parameters) with a single backward pass. For the full 3-PG parameter set (~30 parameters), this is a substantial speedup.

### 11.2 Gradient-based calibration

Given observed stem biomass $W_S^{\text{obs}}$ at $T$ time points, we minimise the MSE loss:

$$\mathcal{L}(\theta) = \frac{1}{T}\sum_{t=1}^{T} \left(W_S^{\text{pred}}(t; \theta) - W_S^{\text{obs}}(t)\right)^2$$

Parameters are transformed for unconstrained optimisation: positivity via log-transform, boundedness via sigmoid:

```python
def loss_fn(log_params_arr, fixed_params, s0, climate, site, obs_WS, obs_times):
    """MSE loss for gradient-based calibration.
    
    log_params_arr: [log(alpha_cx), log(k_D), logit(Y)]
    """
    alpha_cx = jnp.exp(log_params_arr[0])
    k_D      = jnp.exp(log_params_arr[1])
    Y        = jax.nn.sigmoid(log_params_arr[2])

    params = fixed_params._replace(alpha_cx=alpha_cx, k_D=k_D, Y=Y)
    _, outputs = run_3pg(s0, climate, params, site)

    pred_WS = outputs["WS"][obs_times]
    return jnp.mean((pred_WS - obs_WS) ** 2)

# Gradient descent loop
grad_loss = jit(grad(loss_fn))
log_params = jnp.array([jnp.log(0.045), jnp.log(1.5), 0.0])

for step in range(200):
    g = grad_loss(log_params, params, s0, climate, site, obs_WS, obs_times)
    log_params = log_params - 0.01 * g
```

In practice, one would use a proper optimiser (L-BFGS via `jaxopt`, or Adam via `optax`) and calibrate against multiple output variables (LAI, DBH, volume).

### 11.3 Spatial runs via `jax.vmap`

To run 3-PG across a landscape (e.g., varying fertility), `vmap` vectorises the simulation without writing explicit batch loops:

```python
def run_for_FR(FR):
    s = SiteData(latitude=51.8, FR=FR, ASW_max=200.0)
    _, out = run_3pg(initial_state, climate, params, s)
    return out["WS"][-1]

# Run 5 sites in parallel on a single accelerator
FR_values = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
final_WS_by_FR = vmap(run_for_FR)(FR_values)
```

---

## 12. Example Parameters: *Picea abies* (Norway Spruce)

The following values are approximate, loosely based on the r3PG parameter database (Trotsiuk et al., 2020) and the Solling flux site in Germany. **For research use, calibrate against local data.**

| Parameter | Value | Unit | Biological meaning |
|-----------|-------|------|--------------------|
| `alpha_cx` | 0.045 | mol C / mol PAR | Max light-use efficiency |
| `k` | 0.5 | — | Canopy light extinction |
| `Y` | 0.47 | — | Carbon use efficiency |
| `T_min / T_opt / T_max` | 0 / 15 / 35 | °C | Cardinal temperatures |
| `k_F` | 1.0 | — | Frost sensitivity |
| `k_D` | 1.5 | kPa⁻¹ | Stomatal VPD response |
| `fN0` | 0.6 | — | Nutrition modifier at FR=0 |
| `c_theta` | 0.7 | — | Soil water threshold |
| `MaxAge` | 300 | years | Maximum stand age |
| `pFS2 / pFS20` | 1.0 / 0.15 | — | Foliage:stem ratios |
| `r_x / r_n` | 0.80 / 0.25 | — | Root allocation bounds |
| `gammaF1` | 0.020 | month⁻¹ | Mature litterfall rate |
| `gammaR` | 0.015 | month⁻¹ | Root turnover rate |
| `aS / nS` | 0.06 / 2.4 | — | Stem allometric constants |
| `wSx` | 300 | kg | Self-thinning parameter |
| `SLA0 / SLA1` | 10 / 5 | m²/kg | Specific leaf area |
| `rho` | 0.40 | t/m³ | Basic wood density |
| `g_cx` | 0.012 | m/s | Max canopy conductance |

---

## 13. Where to Go from Here

**Calibration:** The r3PG vignette (Trotsiuk et al., 2020) demonstrates a full Morris screening + Bayesian calibration (DEzs MCMC via `BayesianTools`) on the PROFOUND Solling dataset. A JAX equivalent can use `numpyro` or `blackjax` for MCMC, or `jaxopt` / `optax` for MAP estimation.

**Mixed species:** The 3-PGmix extension (Forrester & Tang, 2016) handles multi-cohort stands. The JAX implementation above extends naturally by adding a species dimension to the state and parameter arrays.

**Hybrid ML:** Recent work integrates 3-PG as a differentiable module inside recurrent neural networks for tree-ring-based calibration (Zhou et al., 2024, *Forest Ecology and Management*). The JAX implementation above is directly usable in such architectures via `equinox` or `flax`.

---

## 14. References

1. Landsberg, J.J. & Waring, R.H. (1997). A generalised model of forest productivity using simplified concepts of radiation-use efficiency, carbon balance and partitioning. *Forest Ecology and Management*, 95(3), 209–228. [doi:10.1016/S0378-1127(97)00026-1](https://doi.org/10.1016/S0378-1127(97)00026-1)
2. Waring, R.H., Landsberg, J.J. & Williams, M. (1998). Net primary production of forests: a constant fraction of gross primary production? *Tree Physiology*, 18(2), 129–134.
3. Sands, P.J. & Landsberg, J.J. (2002). Parameterisation of 3-PG for plantation grown *Eucalyptus globulus*. *Forest Ecology and Management*, 163(1–3), 273–292.
4. Esprey, L.J., Sands, P.J. & Smith, C.W. (2004). Understanding 3-PG using a sensitivity analysis. *Forest Ecology and Management*, 193, 235–250. [doi:10.1016/j.foreco.2004.01.032](https://doi.org/10.1016/j.foreco.2004.01.032)
5. Trotsiuk, V., Hartig, F. & Forrester, D.I. (2020). r3PG – An R package for simulating forest growth using the 3-PG process-based model. *Methods in Ecology and Evolution*, 11(10), 1256–1261. [doi:10.1111/2041-210X.13474](https://doi.org/10.1111/2041-210X.13474)
6. Gupta, R. & Sharma, L.K. (2019). The process-based forest growth model 3-PG for use in forest management: A review. *Ecological Modelling*, 397, 55–73.
7. Forrester, D.I. & Tang, X. (2016). Analysing the spatial and temporal dynamics of species interactions in mixed-species forests using the 3-PG model. *Ecological Modelling*, 319, 233–254.
8. Ryan, M.G. & Yoder, B.J. (1997). Hydraulic limits to tree height and tree growth. *BioScience*, 47(4), 235–242.
9. Brouwer, R. (1963). Some aspects of the equilibrium between overground and underground plant parts. *Jaarboek IBS*, 1963, 31–39.
10. Yoda, K. et al. (1963). Self-thinning in overcrowded pure stands under cultivated and natural conditions. *Journal of Biology, Osaka City University*, 14, 107–129.
11. Landsberg, J.J. & Sands, P. (2011). *Physiological Ecology of Forest Production*. Academic Press. ISBN 978-0-12-374460-9.
