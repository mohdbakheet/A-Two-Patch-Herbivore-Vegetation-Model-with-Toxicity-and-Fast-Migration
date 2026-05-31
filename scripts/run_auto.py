# ============================================================
# run_auto.py
# AUTO-07p continuation for reduced two-patch model
# ============================================================

from auto import run, save

# Set this manually to match PAR(2) in two_patch.f90
beta_value = 2.0

eq = run(
    e="two_patch",

    NDIM=3,
    IPS=1,
    IRS=0,
    ILP=1,
    ICP=[1],

    NTST=50,
    NCOL=4,

    NMX=1200,
    NPR=20,

    DS=-0.005,
    DSMIN=1e-6,
    DSMAX=0.02,

    RL0=0.001,
    RL1=0.250,

    ISP=2,
    ISW=1,
    JAC=0,

    EPSL=1e-7,
    EPSU=1e-7,
    EPSS=1e-5,

    #UZR={1: [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0]}
    UZR={1: [0.20, 0.10, 0.05, 0.01]}
)

save(eq, f"eq_alpha2_beta_{beta_value}")

print("\n==============================")
print("AUTO equilibrium continuation")
print("==============================")
print(eq)

print("\n==============================")
print("Searching for Hopf points")
print("==============================")

hopf_labels = []

for lab in range(1, 500):
    try:
        sol = eq(lab)
        ty = sol["TY name"] if "TY name" in sol else ""
        if "HB" in str(ty):
            hopf_labels.append(lab)
    except Exception:
        pass

if len(hopf_labels) == 0:
    print("No Hopf point was detected on this branch.")
    print("Interpretation:")
    print(f"  For beta = {beta_value} and the scanned alpha2 interval,")
    print("  AUTO did not find a complex-conjugate eigenvalue pair")
    print("  crossing the imaginary axis.")
    print("")
    print("Next steps:")
    print("  1. Try larger beta values.")
    print("  2. Continue in beta at fixed alpha2.")
    print("  3. Use MATCONT as an independent check.")
    print("  4. Do not claim Hopf unless AUTO/MATCONT detects an HB point.")
else:
    print(f"Found Hopf labels: {hopf_labels}")
    for lab in hopf_labels:
        hb = eq(lab)
        print("\nHopf point:")
        print(hb)

print(f"\nSaved result as b.eq_alpha2_beta_{beta_value}, s.eq_alpha2_beta_{beta_value}, d.eq_alpha2_beta_{beta_value}")

# ============================================================
# Continue periodic orbit branch from detected Hopf point
# ============================================================

if len(hopf_labels) > 0:
    hb_label = hopf_labels[0]
    hb = eq(hb_label)

    print("\n==============================")
    print("Branch-switching from Hopf point to periodic orbits")
    print("==============================")

    po = run(
        hb,
        IPS=2,          # periodic orbit continuation
        IRS=hb_label,
        ISW=-1,         # branch switch from Hopf
        ICP=[1, 11],    # continue in alpha2 and period
        NTST=80,
        NCOL=4,
        NMX=300,
        NPR=10,
        DS=0.001,
        DSMIN=1e-7,
        DSMAX=0.01,
        RL0=0.001,
        RL1=0.20,
        SP=["BP0", "LP0"],
        EPSL=1e-7,
        EPSU=1e-7,
        EPSS=1e-5
    )

    save(po, f"po_from_hopf_beta_{beta_value}")

    print("\nPeriodic orbit branch:")
    print(po)
    print(f"Saved periodic orbit branch as b.po_from_hopf_beta_{beta_value}, "
          f"s.po_from_hopf_beta_{beta_value}, d.po_from_hopf_beta_{beta_value}")
          

# ============================================================
# Two-parameter Hopf continuation in (alpha2, beta)
# ============================================================

if len(hopf_labels) > 0:
    hb_label = hopf_labels[0]
    hb = eq(hb_label)

    print("\n==============================")
    print("Continuing Hopf curve in (alpha2, beta)")
    print("==============================")

    hopf_curve = run(
        hb,
        IPS=1,
        IRS=hb_label,
        ISW=2,          # continue codimension-one bifurcation
        ICP=[1, 2],     # alpha2 and beta
        NMX=400,
        NPR=10,
        DS=0.005,
        DSMIN=1e-6,
        DSMAX=0.05,
        RL0=0.001,
        RL1=0.20,
        EPSL=1e-7,
        EPSU=1e-7,
        EPSS=1e-5
    )

    save(hopf_curve, "hopf_curve_alpha2_beta")

    print("\nHopf curve:")
    print(hopf_curve)
    print("Saved Hopf curve as b.hopf_curve_alpha2_beta, "
          "s.hopf_curve_alpha2_beta, d.hopf_curve_alpha2_beta")          