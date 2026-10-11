import io
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import streamlit as st

# =============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Calculador Flash Termodinâmico",
    page_icon="⚡",
    layout="wide",
)

# =============================================================================
# 2. BANCO DE DADOS & CONSTANTES
# =============================================================================
R = 8.314462618  # J mol^-1 K^-1
T_REF_PR = 298.15  # K
P_REF_PR = 1.0e5  # Pa

P_MIN_BAR = 0.01
P_MAX_BAR = 2000.0
T_MAX_K_PR = 2000.0
DT_NUM = 0.02
DP_JT_BAR = 0.01
SAT_REL_TOL = 2.0e-5

FLUIDOS_PR = [
    ["H2", 33.18, 13.00, -0.219, 0.00201588, 28.84],
    ["CH4", 190.564, 45.99, 0.011, 0.01604246, 35.69],
    ["CO2", 304.18, 73.80, 0.224, 0.04400950, 37.14],
    ["C2H6", 305.32, 48.70, 0.099, 0.03006904, 52.49],
    ["C3H8", 369.80, 42.50, 0.152, 0.04409562, 73.60],
]

BANCO_PR = {
    item[0].upper(): {
        "nome": item[0],
        "Tc": item[1],
        "Pc_bar": item[2],
        "Pc": item[2] * 1.0e5,
        "omega": item[3],
        "M": item[4],
        "Cp_ig": item[5],
    }
    for item in FLUIDOS_PR
}

# Constantes Leachman H2
R_L = 8.314472
M_L = 0.00201588
TC_L = 33.145
RHOC_L = 15508.0
PC_L = 1_296_400.0
TTRIPLE_L = 13.957
PTRIPLE_L = 7357.828141607672
TMAX_L = 1000.0

A1_L = -1.4579856475
A2_L = 1.888076782
ALOG_L = 1.5
NPE_L = np.array([1.616, -0.4117, -0.792, 0.758, 1.217])
VPE_L = np.array([531.0, 751.0, 1989.0, 2484.0, 6859.0])

D_L = np.array([1, 4, 1, 1, 2, 2, 3, 1, 3], dtype=float)
L_L = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=float)
N_L = np.array([-6.93643, 0.01, 2.1101, 4.52059, 0.732564,
                -1.34086, 0.130985, -0.777414, 0.351944])
TEXP_L = np.array([0.6844, 1.0, 0.989, 0.489, 0.803,
                   1.1444, 1.409, 1.754, 1.311])

BETA_L = np.array([0.171, 0.2245, 0.1304, 0.2785, 0.3967])
DG_L = np.array([2, 1, 3, 1, 1], dtype=float)
EPS_L = np.array([1.506, 0.156, 1.736, 0.67, 1.662])
ETA_L = np.array([1.685, 0.489, 0.103, 2.506, 1.607])
GAMMA_L = np.array([0.7164, 1.3444, 1.4517, 0.7204, 1.5445])
NG_L = np.array([-0.0211716, 0.0226312, 0.032187, -0.0231752, 0.0557346])
TG_L = np.array([4.187, 5.646, 0.791, 7.249, 2.986])

PSN_L = np.array([-0.5553870929135406, -7.0852972599030295,
                  0.006868538454095541, 0.8295514415153659,
                  -0.08773773693832483, 3.754274235303921])
PST_L = np.array([0.882, 1.072, 1.355, 2.61, 8.191, 1.239])

RHOLN_L = np.array([1.9546584452513824, -5.715534926385584,
                    714.2366728890577, -1666.5447521212081,
                    2698.8598951804593, -12755.576112629806])
RHOLT_L = np.array([0.407, 3.401, 7.095, 8.095, 10.911, 17.353])

RHOVN_L = np.array([-3.036644018774332, -773.9211864728516,
                    1530.809493601263, -3087.995404817007,
                    3472.491545729249, -15649.585518005188])
RHOVT_L = np.array([0.473, 6.618, 7.083, 9.094, 10.577, 19.579])

# =============================================================================
# 3. MÉTODOS AUXILIARES E NUMÉRICOS
# =============================================================================
def C_para_K(T_C): return T_C + 273.15
def K_para_C(T_K): return T_K - 273.15
def _finite(x): return x is not None and np.isfinite(x)

def root_scan(func, lo, hi, n=240, log=False):
    if not (_finite(lo) and _finite(hi)) or hi <= lo:
        raise ValueError("Intervalo inválido para busca de raiz.")
    xs = np.geomspace(lo, hi, n) if (log and lo > 0) else np.linspace(lo, hi, n)
    x_prev = None
    f_prev = None
    for x in xs:
        try:
            fx = func(x)
        except Exception:
            continue
        if not np.isfinite(fx):
            continue
        if abs(fx) < 1.0e-10:
            return x
        if f_prev is not None and fx * f_prev < 0.0:
            return brentq(func, x_prev, x, maxiter=200)
        x_prev = x
        f_prev = fx
    raise ValueError(f"Não foi possível localizar raiz em [{lo:g}, {hi:g}].")

# =============================================================================
# 4. TERMODINÂMICA: PENG-ROBINSON
# =============================================================================
def pr_parametros(T, props):
    Tc, Pc, omega = props["Tc"], props["Pc"], props["omega"]
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * (omega**2)
    sqrt_Tr = np.sqrt(T / Tc)
    g = 1.0 + kappa * (1.0 - sqrt_Tr)
    alpha = g**2
    ac = 0.45724 * (R**2) * (Tc**2) / Pc
    a = ac * alpha
    b = 0.07780 * R * Tc / Pc
    dalpha_dT = -kappa * g / (Tc * sqrt_Tr)
    da_dT = ac * dalpha_dT
    return a, b, da_dT

def pr_roots(T, P, props):
    a, b, _ = pr_parametros(T, props)
    A = a * P / ((R * T)**2)
    B = b * P / (R * T)
    coef = [1.0, -(1.0 - B), A - 3.0 * (B**2) - 2.0 * B, -(A * B - B**2 - B**3)]
    roots = np.roots(coef)
    roots = roots[np.abs(roots.imag) < 1.0e-8].real
    roots = roots[roots > B + 1.0e-12]
    roots.sort()
    return roots, A, B

def pr_lnphi(Z, A, B):
    sqrt2 = np.sqrt(2.0)
    if B < 1.0e-14:
        return 0.0
    return (Z - 1.0 - np.log(Z - B) -
            (A / (2.0 * sqrt2 * B)) * np.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B)))

def pr_psat(T, props):
    Tc, Pc = props["Tc"], props["Pc"]
    if T >= Tc or T <= 0.0:
        return None
    P_scan = np.geomspace(max(10.0, 1.0e-8 * Pc), 0.999999 * Pc, 600)
    prev, prevP = None, None
    for P in P_scan:
        roots, A, B = pr_roots(T, P, props)
        if len(roots) == 3:
            F = pr_lnphi(roots[0], A, B) - pr_lnphi(roots[-1], A, B)
            if prev is not None and F * prev < 0.0:
                return brentq(
                    lambda p: (lambda rr_ab: pr_lnphi(rr_ab[0][0], rr_ab[1], rr_ab[2]) -
                                             pr_lnphi(rr_ab[0][-1], rr_ab[1], rr_ab[2]))(pr_roots(T, p, props)),
                    prevP, P, maxiter=120)
            prev, prevP = F, P
        else:
            prev, prevP = None, None
    return None

def pr_tsat(P, props):
    if P >= props["Pc"] or P <= 0.0:
        return None
    Tc = props["Tc"]
    Tmin = max(1.0, 0.25 * Tc)
    Tmax = 0.99999 * Tc
    try:
        return root_scan(lambda T: (pr_psat(T, props) - P) if pr_psat(T, props) else np.nan, Tmin, Tmax, n=160)
    except Exception:
        return None

def pr_root_for_phase(T, P, props, phase):
    roots, A, B = pr_roots(T, P, props)
    if len(roots) == 0:
        raise ValueError("Nenhuma raiz física da equação Peng-Robinson encontrada.")
    Z = roots[0] if phase == "liquido" else roots[-1]
    return Z, A, B

def pr_departures(T, P, props, phase):
    a, b, da_dT = pr_parametros(T, props)
    Z, _, B = pr_root_for_phase(T, P, props, phase)
    sqrt2 = np.sqrt(2.0)
    log_term = np.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B))
    Hres = R * T * (Z - 1.0) + (T * da_dT - a) / (2.0 * sqrt2 * b) * log_term
    Sres = R * np.log(Z - B) + da_dT / (2.0 * sqrt2 * b) * log_term
    V = Z * R * T / P
    return Hres, Sres, V

def pr_phase_stable(T, P, props):
    if T >= props["Tc"]:
        return "condicao_supercritica" if P >= props["Pc"] else "gas"
    ps = pr_psat(T, props)
    if ps is None:
        return "vapor"
    if abs(P - ps) / ps <= SAT_REL_TOL:
        return "saturacao"
    return "vapor" if P < ps else "liquido"

def pr_raw_hs(T, P, props, phase):
    Cp = props["Cp_ig"]
    Hres, Sres, V = pr_departures(T, P, props, phase)
    h_ig = Cp * (T - T_REF_PR)
    s_ig = Cp * np.log(T / T_REF_PR) - R * np.log(P / P_REF_PR)
    return h_ig + Hres, s_ig + Sres, V

def pr_reference_offsets(props):
    phase = pr_phase_stable(T_REF_PR, P_REF_PR, props)
    if phase == "saturacao":
        phase = "vapor"
    h0, s0, _ = pr_raw_hs(T_REF_PR, P_REF_PR, props, phase)
    return h0, s0

def pr_hs_phase(T, P, props, phase):
    h_raw, s_raw, V = pr_raw_hs(T, P, props, phase)
    h0, s0 = pr_reference_offsets(props)
    h = (h_raw - h0) / props["M"] / 1000.0
    s = (s_raw - s0) / props["M"] / 1000.0
    rho_mass = props["M"] / V
    return h, s, V, rho_mass

def pr_sat_state_T(T, props):
    ps = pr_psat(T, props)
    if ps is None:
        raise ValueError("PR: saturação não disponível para esta temperatura.")
    hL, sL, VL, rhoL = pr_hs_phase(T, ps, props, "liquido")
    hV, sV, VV, rhoV = pr_hs_phase(T, ps, props, "vapor")
    return {
        "T_K": T, "P_Pa": ps,
        "hL": hL, "hV": hV, "sL": sL, "sV": sV,
        "rhoL": rhoL, "rhoV": rhoV,
    }

def pr_sat_state_P(P, props):
    T = pr_tsat(P, props)
    if T is None:
        raise ValueError("PR: saturação não disponível para esta pressão.")
    return pr_sat_state_T(T, props)

def pr_basic_TP(T, P, props, force_phase=None):
    phase = force_phase or pr_phase_stable(T, P, props)
    if phase == "saturacao":
        return {"ambiguous_sat": True, "sat": pr_sat_state_T(T, props)}
    h, s, V, rho = pr_hs_phase(T, P, props, phase)
    return {
        "T_K": T, "P_Pa": P, "H": h, "S": s,
        "V_molar": V, "rho": rho, "phase": phase,
        "x": None, "ambiguous_sat": False,
    }

def pr_cp_mu(T, P, props, phase):
    dT = max(DT_NUM, 1.0e-5 * T)
    hp = pr_hs_phase(T + dT, P, props, phase)[0]
    hm = pr_hs_phase(T - dT, P, props, phase)[0]
    Cp = (hp - hm) / (2.0 * dT)
    Vp = pr_hs_phase(T + dT, P, props, phase)[2]
    Vm = pr_hs_phase(T - dT, P, props, phase)[2]
    V = pr_hs_phase(T, P, props, phase)[2]
    dVdT = (Vp - Vm) / (2.0 * dT)
    Cp_molar = Cp * 1000.0 * props["M"]
    mu_Pa = (T * dVdT - V) / Cp_molar
    return Cp, mu_Pa * 1.0e5

# =============================================================================
# 5. TERMODINÂMICA: LEACHMAN (H2)
# =============================================================================
def l_alpha0(tau, delta):
    x = VPE_L * tau / TC_L
    return (np.log(delta) + A1_L + A2_L * tau + ALOG_L * np.log(tau)
            + np.sum(NPE_L * np.log(1.0 - np.exp(-x))))

def l_alpha0_tau(tau):
    x = VPE_L * tau / TC_L
    return A2_L + ALOG_L / tau + np.sum(NPE_L * (VPE_L / TC_L) * np.exp(-x) / (1.0 - np.exp(-x)))

def l_alphar_derivatives(tau, delta):
    expo_power = np.where(L_L == 0.0, 0.0, -delta**L_L)
    terms = N_L * delta**D_L * tau**TEXP_L * np.exp(expo_power)
    ar = np.sum(terms)
    ar_delta = np.sum(terms * (D_L / delta - np.where(L_L == 0.0, 0.0, L_L * delta**(L_L - 1.0))))
    ar_tau = np.sum(terms * TEXP_L / tau)
    terms_g = NG_L * delta**DG_L * tau**TG_L * np.exp(-ETA_L * (delta - EPS_L)**2 - BETA_L * (tau - GAMMA_L)**2)
    ar += np.sum(terms_g)
    ar_delta += np.sum(terms_g * (DG_L / delta - 2.0 * ETA_L * (delta - EPS_L)))
    ar_tau += np.sum(terms_g * (TG_L / tau - 2.0 * BETA_L * (tau - GAMMA_L)))
    return ar, ar_delta, ar_tau

def l_properties_Trho(T, rho):
    tau, delta = TC_L / T, rho / RHOC_L
    a0, a0_tau = l_alpha0(tau, delta), l_alpha0_tau(tau)
    ar, ar_delta, ar_tau = l_alphar_derivatives(tau, delta)
    P = rho * R_L * T * (1.0 + delta * ar_delta)
    h_molar = R_L * T * (1.0 + tau * (a0_tau + ar_tau) + delta * ar_delta)
    s_molar = R_L * (tau * (a0_tau + ar_tau) - a0 - ar)
    return P, h_molar, s_molar

def l_sat_ancillary(T):
    if not (TTRIPLE_L <= T < TC_L):
        raise ValueError("Leachman: temperatura fora da região líquido-vapor.")
    theta = 1.0 - T / TC_L
    Psat = PC_L * np.exp(TC_L / T * np.sum(PSN_L * theta**PST_L))
    rhoL = RHOC_L * (1.0 + np.sum(RHOLN_L * theta**RHOLT_L))
    rhoV = RHOC_L * np.exp(TC_L / T * np.sum(RHOVN_L * theta**RHOVT_L))
    return Psat, rhoL, rhoV

def l_sat_state_T(T):
    Psat, rhoL, rhoV = l_sat_ancillary(T)
    _, hL, sL = l_properties_Trho(T, rhoL)
    _, hV, sV = l_properties_Trho(T, rhoV)
    return {
        "T_K": T, "P_Pa": Psat,
        "hL": hL / M_L / 1000.0, "hV": hV / M_L / 1000.0,
        "sL": sL / M_L / 1000.0, "sV": sV / M_L / 1000.0,
        "rhoL": rhoL * M_L, "rhoV": rhoV * M_L,
    }

def l_tsat(P):
    if P < PTRIPLE_L or P >= PC_L:
        return None
    try:
        return brentq(lambda T: l_sat_ancillary(T)[0] - P, TTRIPLE_L, TC_L * (1.0 - 1.0e-10), maxiter=150)
    except Exception:
        return None

def l_sat_state_P(P):
    T = l_tsat(P)
    if T is None:
        raise ValueError("Leachman: saturação não disponível para esta pressão.")
    return l_sat_state_T(T)

def l_solve_density_phase(T, P, phase=None):
    if T <= 0.0 or T > TMAX_L:
        raise ValueError("Leachman: temperatura fora da faixa permitida.")
    def f(rho): return l_properties_Trho(T, rho)[0] - P
    if T < TC_L and T >= TTRIPLE_L:
        Psat, rhoLsat, rhoVsat = l_sat_ancillary(T)
        if phase is None:
            phase = "vapor" if P < Psat else "liquido"
        if phase == "vapor":
            lo = max(1.0e-12, P / (R_L * T) * 1.0e-6)
            hi = max(rhoVsat * 1.000001, lo * 10.0)
            return brentq(f, lo, hi, maxiter=200)
        if phase == "liquido":
            lo = rhoLsat * 0.999999
            hi = 120000.0
            return brentq(f, lo, hi, maxiter=200)
    lo = max(1.0e-12, P / (R_L * T) * 1.0e-7)
    hi = 120000.0
    return brentq(f, lo, hi, maxiter=250)

def l_phase_stable(T, P):
    if T >= TC_L:
        return "condicao_supercritica" if P >= PC_L else "gas"
    if T < TTRIPLE_L:
        raise ValueError("Leachman: abaixo do ponto triplo.")
    ps = l_sat_ancillary(T)[0]
    if abs(P - ps) / ps <= SAT_REL_TOL:
        return "saturacao"
    return "vapor" if P < ps else "liquido"

def l_hs_phase(T, P, phase):
    rho_molar = l_solve_density_phase(T, P, phase)
    _, h, s = l_properties_Trho(T, rho_molar)
    return h / M_L / 1000.0, s / M_L / 1000.0, 1.0 / rho_molar, rho_molar * M_L

def l_basic_TP(T, P, force_phase=None):
    phase = force_phase or l_phase_stable(T, P)
    if phase == "saturacao":
        return {"ambiguous_sat": True, "sat": l_sat_state_T(T)}
    h, s, V, rho = l_hs_phase(T, P, phase)
    return {
        "T_K": T, "P_Pa": P, "H": h, "S": s,
        "V_molar": V, "rho": rho, "phase": phase,
        "x": None, "ambiguous_sat": False,
    }

def l_cp_mu(T, P, phase):
    dT = max(DT_NUM, 1.0e-5 * T)
    hp = l_hs_phase(T + dT, P, phase)[0]
    hm = l_hs_phase(T - dT, P, phase)[0]
    Cp = (hp - hm) / (2.0 * dT)
    Vp = l_hs_phase(T + dT, P, phase)[2]
    Vm = l_hs_phase(T - dT, P, phase)[2]
    V = l_hs_phase(T, P, phase)[2]
    dVdT = (Vp - Vm) / (2.0 * dT)
    Cp_molar = Cp * 1000.0 * M_L
    mu_Pa = (T * dVdT - V) / Cp_molar
    return Cp, mu_Pa * 1.0e5

# =============================================================================
# 6. RESOLUÇÃO COMUM & REGRA DA ALAVANCA
# =============================================================================
def get_model_data(modelo, fluido):
    if modelo == "PR":
        return modelo, BANCO_PR[fluido]
    if modelo == "LEACHMAN":
        return modelo, None
    raise ValueError("Modelo não suportado.")

def model_limits(modelo, props):
    if modelo == "LEACHMAN":
        return TTRIPLE_L + 1.0e-5, TMAX_L, P_MIN_BAR * 1e5, P_MAX_BAR * 1e5
    return max(1.0, 0.25 * props["Tc"]), T_MAX_K_PR, P_MIN_BAR * 1e5, P_MAX_BAR * 1e5

def model_Tc_Pc(modelo, props):
    return (TC_L, PC_L) if modelo == "LEACHMAN" else (props["Tc"], props["Pc"])

def sat_T(modelo, T, props):
    return l_sat_state_T(T) if modelo == "LEACHMAN" else pr_sat_state_T(T, props)

def sat_P(modelo, P, props):
    return l_sat_state_P(P) if modelo == "LEACHMAN" else pr_sat_state_P(P, props)

def basic_TP(modelo, T, P, props, force_phase=None):
    return l_basic_TP(T, P, force_phase) if modelo == "LEACHMAN" else pr_basic_TP(T, P, props, force_phase)

def cp_mu_TP(modelo, T, P, props, phase):
    return l_cp_mu(T, P, phase) if modelo == "LEACHMAN" else pr_cp_mu(T, P, props, phase)

def pressure_Trho(modelo, T, rho_mass, props):
    if rho_mass <= 0.0:
        raise ValueError("Densidade deve ser positiva.")
    if modelo == "LEACHMAN":
        return l_properties_Trho(T, rho_mass / M_L)[0]
    M, V = props["M"], props["M"] / rho_mass
    a, b, _ = pr_parametros(T, props)
    if V <= b:
        raise ValueError("PR: densidade resulta em volume não físico (V <= b).")
    return R * T / (V - b) - a / (V * (V + b) + b * (V - b))

def quality_from_density(rho, rhoL, rhoV):
    v, vL, vV = 1.0 / rho, 1.0 / rhoL, 1.0 / rhoV
    if abs(vV - vL) < 1.0e-30:
        raise ValueError("vV e vL são coincidentes.")
    return (v - vL) / (vV - vL)

def two_phase_from_sat(sat, x, specified=None):
    x = float(np.clip(x, 0.0, 1.0))
    H = sat["hL"] + x * (sat["hV"] - sat["hL"])
    S = sat["sL"] + x * (sat["sV"] - sat["sL"])
    rho = 1.0 / ((1.0 - x) / sat["rhoL"] + x / sat["rhoV"])
    if x <= 1.0e-10:
        phase = "liquido_saturado"
    elif x >= 1.0 - 1.0e-10:
        phase = "vapor_saturado"
    else:
        phase = "bifasico"
    return {
        "T_K": sat["T_K"], "P_Pa": sat["P_Pa"], "H": H, "S": S,
        "rho": rho, "phase": phase, "x": x,
        "ambiguous_sat": False, "specified": specified, "sat": sat,
    }

def flash_core(modelo, fluido, par, valor1, valor2):
    modelo, props = get_model_data(modelo, fluido)
    par = par.upper()
    Tmin, Tmax, Pmin, Pmax = model_limits(modelo, props)
    Tc, Pc = model_Tc_Pc(modelo, props)

    if par == "TP":
        T, P = C_para_K(valor1), valor2 * 1.0e5
        if not (Tmin <= T <= Tmax) or not (Pmin <= P <= Pmax):
            raise ValueError("Valores fora dos limites do modelo.")
        st = basic_TP(modelo, T, P, props)
        st["specified"] = "T"
        return st

    if par == "TRHO":
        T, rho_target = C_para_K(valor1), valor2
        if not (Tmin <= T <= Tmax) or rho_target <= 0.0:
            raise ValueError("Valores inválidos para T ou densidade.")
        sat, force_phase = None, None
        if T < Tc:
            try: sat = sat_T(modelo, T, props)
            except Exception: sat = None
        if sat is not None:
            rhoL, rhoV = sat["rhoL"], sat["rhoV"]
            tol_rho = 1.0e-9 * max(1.0, abs(rhoL), abs(rhoV))
            if rhoV - tol_rho <= rho_target <= rhoL + tol_rho:
                st = two_phase_from_sat(sat, quality_from_density(rho_target, rhoL, rhoV), specified="T")
                st["rho"] = rho_target
                return st
            force_phase = "vapor" if rho_target < rhoV else "liquido"
        P = pressure_Trho(modelo, T, rho_target, props)
        st = basic_TP(modelo, T, P, props, force_phase=force_phase)
        st["specified"] = "T"
        st["rho"] = rho_target
        return st

    if par == "PRHO":
        P, rho_target = valor1 * 1.0e5, valor2
        if not (Pmin <= P <= Pmax) or rho_target <= 0.0:
            raise ValueError("Valores inválidos para P ou densidade.")
        sat = None
        if P < Pc:
            try: sat = sat_P(modelo, P, props)
            except Exception: sat = None
        if sat is not None:
            rhoL, rhoV = sat["rhoL"], sat["rhoV"]
            tol_rho = 1.0e-9 * max(1.0, abs(rhoL), abs(rhoV))
            if rhoV - tol_rho <= rho_target <= rhoL + tol_rho:
                st = two_phase_from_sat(sat, quality_from_density(rho_target, rhoL, rhoV), specified="P")
                st["rho"] = rho_target
                return st
            if rho_target < rhoV:
                Tlo, Thi, force_phase = sat["T_K"] * (1.0 + 2.0e-7), Tmax, "vapor"
            else:
                Tlo, Thi, force_phase = Tmin, sat["T_K"] * (1.0 - 2.0e-7), "liquido"
            Tsol = root_scan(lambda T: pressure_Trho(modelo, T, rho_target, props) - P, Tlo, Thi, n=360)
            st = basic_TP(modelo, Tsol, P, props, force_phase=force_phase)
            st["specified"] = "P"
            st["rho"] = rho_target
            return st
        Tsol = root_scan(lambda T: pressure_Trho(modelo, T, rho_target, props) - P, Tmin, Tmax, n=500)
        st = basic_TP(modelo, Tsol, P, props)
        st["specified"] = "P"
        st["rho"] = rho_target
        return st

    if par in {"PH", "PS"}:
        P, target = valor1 * 1.0e5, valor2
        key = "H" if par == "PH" else "S"
        sat = None
        if P < Pc:
            try: sat = sat_P(modelo, P, props)
            except Exception: sat = None
        if sat is not None:
            low, high = sat["hL"] if key == "H" else sat["sL"], sat["hV"] if key == "H" else sat["sV"]
            tol = 1.0e-9 * max(1.0, abs(low), abs(high))
            if low - tol <= target <= high + tol:
                st = two_phase_from_sat(sat, (target - low) / (high - low), specified="P")
                st[key] = target
                return st
            if target < low:
                Tlo, Thi, force_phase = Tmin, sat["T_K"] * (1.0 - 2.0e-7), "liquido"
            else:
                Tlo, Thi, force_phase = sat["T_K"] * (1.0 + 2.0e-7), Tmax, "vapor"
            Tsol = root_scan(lambda T: basic_TP(modelo, T, P, props, force_phase=force_phase)[key] - target, Tlo, Thi, n=280)
            st = basic_TP(modelo, Tsol, P, props, force_phase=force_phase)
            st["specified"] = "P"
            st[key] = target
            return st
        Tsol = root_scan(lambda T: basic_TP(modelo, T, P, props)[key] - target, Tmin, Tmax, n=320)
        st = basic_TP(modelo, Tsol, P, props)
        st["specified"] = "P"
        st[key] = target
        return st

    if par in {"TH", "TS"}:
        T, target = C_para_K(valor1), valor2
        key = "H" if par == "TH" else "S"
        sat = None
        if T < Tc:
            try: sat = sat_T(modelo, T, props)
            except Exception: sat = None
        if sat is not None:
            low, high = sat["hL"] if key == "H" else sat["sL"], sat["hV"] if key == "H" else sat["sV"]
            tol = 1.0e-9 * max(1.0, abs(low), abs(high))
            if low - tol <= target <= high + tol:
                st = two_phase_from_sat(sat, (target - low) / (high - low), specified="T")
                st[key] = target
                return st
            if target < low:
                Plo, Phi, force_phase = sat["P_Pa"] * (1.0 + 2.0e-7), Pmax, "liquido"
            else:
                Plo, Phi, force_phase = Pmin, sat["P_Pa"] * (1.0 - 2.0e-7), "vapor"
            Psol = root_scan(lambda P: basic_TP(modelo, T, P, props, force_phase=force_phase)[key] - target, Plo, Phi, n=320, log=True)
            st = basic_TP(modelo, T, Psol, props, force_phase=force_phase)
            st["specified"] = "T"
            st[key] = target
            return st
        Psol = root_scan(lambda P: basic_TP(modelo, T, P, props)[key] - target, Pmin, Pmax, n=360, log=True)
        st = basic_TP(modelo, T, Psol, props)
        st["specified"] = "T"
        st[key] = target
        return st

    raise ValueError("Par de entrada desconhecido.")

def muJT_two_phase(modelo, fluido, state):
    P0_bar, H0 = state["P_Pa"] / 1.0e5, state["H"]
    dP = max(DP_JT_BAR, 1.0e-4 * max(1.0, P0_bar))
    Pminus, Pplus = max(P_MIN_BAR, P0_bar - dP), min(P_MAX_BAR, P0_bar + dP)
    if Pplus <= Pminus:
        return np.nan
    try:
        sm = flash_core(modelo, fluido, "PH", Pminus, H0)
        sp = flash_core(modelo, fluido, "PH", Pplus, H0)
        return (sp["T_K"] - sm["T_K"]) / (Pplus - Pminus)
    except Exception:
        return np.nan

def augment_properties(modelo, fluido, state):
    if state.get("ambiguous_sat", False):
        state["Cp"], state["muJT"] = np.nan, np.nan
        return state
    phase = state["phase"]
    if phase in {"bifasico", "liquido_saturado", "vapor_saturado"}:
        state["Cp"] = np.inf if phase == "bifasico" else np.nan
        state["muJT"] = muJT_two_phase(modelo, fluido, state)
        return state
    modelo_u, props = get_model_data(modelo, fluido)
    Cp, mu = cp_mu_TP(modelo_u, state["T_K"], state["P_Pa"], props, phase)
    state["Cp"], state["muJT"] = Cp, mu
    return state

def calcular_estado(modelo, fluido, par, v1, v2):
    st_res = flash_core(modelo, fluido, par, v1, v2)
    return augment_properties(modelo, fluido, st_res)

def gerar_texto_relatorio(st, modelo, fluido, par, valor1, valor2):
    nomes_entrada = {
        "TP": ("Temperatura", "Pressão"), "TRHO": ("Temperatura", "Densidade"),
        "PRHO": ("Pressão", "Densidade"), "PH": ("Pressão", "Entalpia específica"),
        "TH": ("Temperatura", "Entalpia específica"), "PS": ("Pressão", "Entropia específica"),
        "TS": ("Temperatura", "Entropia específica"),
    }
    unidades_entrada = {
        "TP": ("°C", "bar"), "TRHO": ("°C", "kg/m3"), "PRHO": ("bar", "kg/m3"),
        "PH": ("bar", "kJ/kg"), "TH": ("°C", "kJ/kg"), "PS": ("bar", "kJ/(kg K)"),
        "TS": ("°C", "kJ/(kg K)"),
    }
    nome1, nome2 = nomes_entrada[par]
    un1, un2 = unidades_entrada[par]
    ref_txt = "H = S = 0 em 25 °C e 1 bar" if modelo == "PR" else "Nativa da formulação de Leachman"

    linhas = [
        "=" * 78,
        "RELATÓRIO DE PROPRIEDADES TERMODINÂMICAS - FLUIDO PURO",
        "=" * 78,
        f"Modelo termodinâmico : {modelo}",
        f"Fluido               : {fluido}",
        f"Par de entrada       : {par}",
        f"Referência de H e S   : {ref_txt}",
        "",
        "VARIÁVEIS ESPECIFICADAS",
        "-" * 78,
        f"{nome1:<26s}: {valor1:.10g} {un1}",
        f"{nome2:<26s}: {valor2:.10g} {un2}",
        "",
    ]
    if st.get("ambiguous_sat", False):
        sat = st["sat"]
        linhas += [
            "ESTADO TERMODINÂMICO", "-" * 78,
            "Fase / condição          : saturação subespecificada",
            "Observação              : T e P sobre a curva não determinam o título x.",
            "",
            "PROPRIEDADES DE SATURAÇÃO", "-" * 78,
            f"{'Temperatura':<26s}: {K_para_C(sat['T_K']):.8f} °C",
            f"{'Pressão':<26s}: {sat['P_Pa']/1e5:.8f} bar",
            f"{'H líquido saturado':<26s}: {sat['hL']:.8f} kJ/kg",
            f"{'H vapor saturado':<26s}: {sat['hV']:.8f} kJ/kg",
            f"{'S líquido saturado':<26s}: {sat['sL']:.8f} kJ/(kg K)",
            f"{'S vapor saturado':<26s}: {sat['sV']:.8f} kJ/(kg K)",
            f"{'rho líquido saturado':<26s}: {sat['rhoL']:.8f} kg/m3",
            f"{'rho vapor saturado':<26s}: {sat['rhoV']:.8f} kg/m3",
            f"{'Cp':<26s}: não definido",
            f"{'mu_JT':<26s}: não definido",
            "=" * 78,
        ]
        return "\n".join(linhas)

    linhas += [
        "RESULTADOS CALCULADOS", "-" * 78,
        f"{'Fase':<26s}: {st['phase']}",
    ]
    if st.get("x") is not None:
        linhas.append(f"{'Título, x':<26s}: {st['x']:.10f}")
    linhas += [
        f"{'Temperatura':<26s}: {K_para_C(st['T_K']):.8f} °C",
        f"{'Pressão':<26s}: {st['P_Pa']/1e5:.8f} bar",
        f"{'Entalpia específica, H':<26s}: {st['H']:.8f} kJ/kg",
        f"{'Entropia específica, S':<26s}: {st['S']:.8f} kJ/(kg K)",
    ]
    if np.isinf(st["Cp"]):
        linhas.append(f"{'Cp':<26s}: infinito / não finito na região bifásica")
    elif np.isnan(st["Cp"]):
        linhas.append(f"{'Cp':<26s}: não disponível")
    else:
        linhas.append(f"{'Cp':<26s}: {st['Cp']:.8f} kJ/(kg K)")

    if np.isnan(st["muJT"]):
        linhas.append(f"{'Coef. Joule-Thomson':<26s}: não disponível")
    else:
        linhas.append(f"{'Coef. Joule-Thomson':<26s}: {st['muJT']:.10f} K/bar")

    if "rho" in st and _finite(st["rho"]):
        linhas.append(f"{'Densidade':<26s}: {st['rho']:.8f} kg/m3")

    linhas += ["", "=" * 78, "FIM DO RELATÓRIO", "=" * 78]
    return "\n".join(linhas)

# =============================================================================
# 7. INTERFACE STREAMLIT
# =============================================================================
st.title("⚡ Flash & Calculador de Propriedades de Fluido Puro")
st.markdown("Cálculo termodinâmico rigoroso utilizando as equações **Peng-Robinson** ou **Leachman (H2)**.")

# --- SIDEBAR: ENTRADAS ---
with st.sidebar:
    st.header("Configuração do Flash")
    modelo_opt = st.selectbox("Modelo Termodinâmico", ["LEACHMAN", "PR"], index=0)

    if modelo_opt == "LEACHMAN":
        fluido_opt = st.selectbox("Fluido", ["H2"], index=0)
        st.info("Leachman: formulação de Helmholtz dedicada ao H₂ normal.")
    else:
        lista_pr = list(BANCO_PR.keys())
        fluido_opt = st.selectbox("Fluido", lista_pr, index=0)
        st.info("Peng-Robinson: estado padrão H = 0 e S = 0 a 25 °C e 1 bar.")

    pares_dict = {
        "TP": ("Temperatura [°C]", "Pressão [bar]"),
        "TRHO": ("Temperatura [°C]", "Densidade [kg/m³]"),
        "PRHO": ("Pressão [bar]", "Densidade [kg/m³]"),
        "PH": ("Pressão [bar]", "Entalpia [kJ/kg]"),
        "TH": ("Temperatura [°C]", "Entalpia [kJ/kg]"),
        "PS": ("Pressão [bar]", "Entropia [kJ/(kg K)]"),
        "TS": ("Temperatura [°C]", "Entropia [kJ/(kg K)]"),
    }

    par_opt = st.selectbox("Par de Entrada", list(pares_dict.keys()), index=0)
    lbl1, lbl2 = pares_dict[par_opt]

    # Valores default para TP padrão
    def_v1 = 25.0 if par_opt.startswith("T") else 10.0
    def_v2 = 150.0 if "P" in par_opt and not par_opt.startswith("P") else 20.0

    val1 = st.number_input(lbl1, value=float(def_v1), format="%.4f")
    val2 = st.number_input(lbl2, value=float(def_v2), format="%.4f")

    calcular_btn = st.button("Calcular Estado", type="primary", use_container_width=True)

# --- EXECUÇÃO E APRESENTAÇÃO ---
if calcular_btn:
    try:
        with st.spinner("Resolvendo equilíbrio e calculando derivadas..."):
            estado = calcular_estado(modelo_opt, fluido_opt, par_opt, val1, val2)

        st.subheader("Resultados Termodinâmicos")

        if estado.get("ambiguous_sat", False):
            st.warning("⚠️ **Saturação Subespecificada:** O par (T, P) coincide com a curva de equilíbrio líquido-vapor. Em um sistema monocomponente, a regra das fases confere 1 grau de liberdade neste ponto, tornando impossível determinar o título $x$ apenas por T e P.")
            sat = estado["sat"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("T de saturação", f"{K_para_C(sat['T_K']):.4f} °C")
                st.metric("H Líquido Saturado", f"{sat['hL']:.2f} kJ/kg")
                st.metric("S Líquido Saturado", f"{sat['sL']:.4f} kJ/(kg K)")
                st.metric("Densidade Líquido", f"{sat['rhoL']:.2f} kg/m³")
            with col2:
                st.metric("P de saturação", f"{sat['P_Pa']/1e5:.4f} bar")
                st.metric("H Vapor Saturado", f"{sat['hV']:.2f} kJ/kg")
                st.metric("S Vapor Saturado", f"{sat['sV']:.4f} kJ/(kg K)")
                st.metric("Densidade Vapor", f"{sat['rhoV']:.4f} kg/m³")
        else:
            # Exibição de cards
            col_fase, col_x, col_t, col_p = st.columns(4)
            col_fase.metric("Fase", estado["phase"].replace("_", " ").title())
            col_x.metric("Título (x)", f"{estado['x']:.4f}" if estado.get("x") is not None else "N/A")
            col_t.metric("Temperatura", f"{K_para_C(estado['T_K']):.4f} °C")
            col_p.metric("Pressão", f"{estado['P_Pa']/1e5:.4f} bar")

            col_h, col_s, col_rho, col_cp, col_mu = st.columns(5)
            col_h.metric("Entalpia (H)", f"{estado['H']:.2f} kJ/kg")
            col_s.metric("Entropia (S)", f"{estado['S']:.4f} kJ/(kg K)")
            col_rho.metric("Densidade (ρ)", f"{estado['rho']:.3f} kg/m³" if _finite(estado.get("rho")) else "N/A")

            cp_str = "∞ (Bifásico)" if np.isinf(estado["Cp"]) else (f"{estado['Cp']:.4f} kJ/(kg K)" if _finite(estado["Cp"]) else "N/A")
            col_cp.metric("Cp", cp_str)
            col_mu.metric("μ_JT", f"{estado['muJT']:.5f} K/bar" if _finite(estado["muJT"]) else "N/A")

            # Detalhes de envelope se houver saturação associada
            if "sat" in estado and isinstance(estado["sat"], dict):
                with st.expander("Ver Limites de Saturação da Condição"):
                    sat = estado["sat"]
                    dados_sat = {
                        "Propriedade": ["Temperatura (°C)", "Pressão (bar)", "Entalpia (kJ/kg)", "Entropia (kJ/(kg K))", "Densidade (kg/m³)"],
                        "Líquido Saturado": [f"{K_para_C(sat['T_K']):.4f}", f"{sat['P_Pa']/1e5:.4f}", f"{sat['hL']:.2f}", f"{sat['sL']:.4f}", f"{sat['rhoL']:.3f}"],
                        "Vapor Saturado": [f"{K_para_C(sat['T_K']):.4f}", f"{sat['P_Pa']/1e5:.4f}", f"{sat['hV']:.2f}", f"{sat['sV']:.4f}", f"{sat['rhoV']:.3f}"]
                    }
                    st.table(dados_sat)

        # Botão de download do relatório .txt
        relatorio_conteudo = gerar_texto_relatorio(estado, modelo_opt, fluido_opt, par_opt, val1, val2)
        st.download_button(
            label="📄 Baixar Relatório em .txt",
            data=relatorio_conteudo,
            file_name=f"relatorio_{modelo_opt}_{fluido_opt}_{par_opt}.txt",
            mime="text/plain",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Erro durante o cálculo termodinâmico: {e}")
else:
    st.info("Selecione os parâmetros na barra lateral e clique em **Calcular Estado**.")
