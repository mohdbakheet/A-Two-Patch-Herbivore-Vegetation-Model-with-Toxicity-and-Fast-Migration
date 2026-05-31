! ============================================================
! two_patch.f90
! AUTO-07p file for reduced two-patch herbivore-vegetation model
!
! State variables:
!   U(1) = v1
!   U(2) = v2
!   U(3) = H
!
! Parameters:
!   PAR(1) = alpha2_dim
!   PAR(2) = beta
! ============================================================

SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: NDIM, IJAC
  INTEGER, INTENT(IN) :: ICP(*)
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
  DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
  DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

  DOUBLE PRECISION :: v1, v2, H
  DOUBLE PRECISION :: Kone, Ktwo, r, c0, imax, vu, b
  DOUBLE PRECISION :: mu1_dim, G, alpha1_dim
  DOUBLE PRECISION :: b1, b2, eta, conv, alpha1, alpha2
  DOUBLE PRECISION :: kap1, kap2, mu, beta, rho1, rho2
  DOUBLE PRECISION :: den1, den2, C1, C2
  DOUBLE PRECISION :: m1, m2, d, hstar
  DOUBLE PRECISION :: grazing1, grazing2

  ! ----------------------------------------------------------
  ! State variables
  ! ----------------------------------------------------------
  v1 = U(1)
  v2 = U(2)
  H  = U(3)

  ! ----------------------------------------------------------
  ! Dimensional baseline parameters
  ! These match the Python coexistence finder
  ! ----------------------------------------------------------
  Kone       = 150.0D0
  Ktwo       = 100.0D0
  r          = 0.2D0
  c0         = 0.175D0
  imax       = 0.75D0
  vu         = 10.0D0
  b          = 20.0D0

  mu1_dim    = 0.003D0
  G          = 1.10D0
  alpha1_dim = 0.35D0

  ! ----------------------------------------------------------
  ! Nondimensional parameters
  ! ----------------------------------------------------------
  b1     = b / Kone
  b2     = b / Ktwo
  eta    = imax / r
  conv   = c0 * imax / r

  alpha1 = alpha1_dim / r
  alpha2 = PAR(1) / r

  kap1   = 1.0D0 / Kone
  kap2   = 1.0D0 / Ktwo

  mu     = mu1_dim / r
  beta   = PAR(2)

  rho1   = vu / Kone
  rho2   = vu / Ktwo

  ! ----------------------------------------------------------
  ! Denominators
  ! IMPORTANT:
  ! For continuation, avoid max(), min(), or clipping.
  ! We assume continuation remains in v1 > rho1 and v2 > rho2.
  ! ----------------------------------------------------------
  den1 = b1 + v1 - rho1
  den2 = b2 + v2 - rho2

  ! ----------------------------------------------------------
  ! Toxicity modifiers
  ! C_i(v_i) = 1 - beta * (v_i-rho_i)/(b_i+v_i-rho_i)
  ! ----------------------------------------------------------
  C1 = 1.0D0 - beta * (v1 - rho1) / den1
  C2 = 1.0D0 - beta * (v2 - rho2) / den2

  ! ----------------------------------------------------------
  ! Fast migration rates and fast equilibrium h*
  ! ----------------------------------------------------------
  m1 = alpha1 / (kap1 + v1)
  m2 = alpha2 / (kap2 + v2)
  d  = m1 + m2

  hstar = (m2 / d) * H

  ! ----------------------------------------------------------
  ! Grazing terms
  ! ----------------------------------------------------------
  grazing1 = eta * (v1 - rho1) * hstar / den1 * C1
  grazing2 = eta * (v2 - rho2) * (H - hstar) / den2 * C2

  ! ----------------------------------------------------------
  ! Reduced model
  ! ----------------------------------------------------------
  F(1) = v1 * (1.0D0 - v1) - grazing1

  F(2) = v2 * (1.0D0 - v2) - grazing2

  F(3) = C1 * conv * (v1 - rho1) * hstar / den1 &
       + C2 * conv * (v2 - rho2) * (H - hstar) / den2 &
       - mu * H

  RETURN
END SUBROUTINE FUNC


SUBROUTINE STPNT(NDIM,U,PAR,T)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: NDIM
  DOUBLE PRECISION, INTENT(OUT) :: U(NDIM), PAR(*)
  DOUBLE PRECISION, INTENT(IN) :: T

  PAR(1) = 0.25D0
  PAR(2) = 2.0D0

  U(1) = 0.07091148007292154D0
  U(2) = 0.1042408816925011D0
  U(3) = 1.858007227338037D0

  RETURN
END SUBROUTINE STPNT


SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: NDIM, NBC, IJAC
  INTEGER, INTENT(IN) :: ICP(*)
  DOUBLE PRECISION, INTENT(IN) :: PAR(*), U0(NDIM), U1(NDIM)
  DOUBLE PRECISION, INTENT(OUT) :: FB(NBC)
  DOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)

  RETURN
END SUBROUTINE BCND


SUBROUTINE ICND(NDIM,PAR,ICP,NINT,U,UOLD,UDOT,UPOLD,FI,IJAC,DINT)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: NDIM, NINT, IJAC
  INTEGER, INTENT(IN) :: ICP(*)
  DOUBLE PRECISION, INTENT(IN) :: PAR(*), U(NDIM), UOLD(NDIM), UDOT(NDIM), UPOLD(NDIM)
  DOUBLE PRECISION, INTENT(OUT) :: FI(NINT)
  DOUBLE PRECISION, INTENT(INOUT) :: DINT(NINT,*)

  RETURN
END SUBROUTINE ICND


SUBROUTINE FOPT(NDIM,U,ICP,PAR,IJAC,FS,DFDU,DFDP)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: NDIM, IJAC
  INTEGER, INTENT(IN) :: ICP(*)
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
  DOUBLE PRECISION, INTENT(OUT) :: FS
  DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM), DFDP(*)

  FS = 0.0D0

  RETURN
END SUBROUTINE FOPT


SUBROUTINE PVLS(NDIM,U,PAR)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: NDIM
  DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
  DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)

  ! Optional diagnostic parameters can be stored here.
  ! For now, leave unused.

  RETURN
END SUBROUTINE PVLS