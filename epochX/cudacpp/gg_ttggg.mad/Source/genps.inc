c*************************************************************************
c       Parameters used by genps and dsample, you must recompile
c	dsample if you change anything below
c*************************************************************************
      include 'maxparticles.inc'
      integer    max_branch
      parameter (max_branch=max_particles-1)
      integer    ng     , maxdim     , maxinvar     , maxconfigs
      parameter (ng = 96, maxdim = 3*(max_particles-2)-1, maxinvar= 4*max_particles, maxconfigs=10)
c      parameter (ng = 43, maxdim = 25, maxinvar= 57 , maxconfigs=20)

      double precision  xgmin,     xgmax
      parameter        (xgmin=-1d0, xgmax=1d0)

      integer    maxevents         !Requires about 1K/event
      parameter (maxevents=200000)  !Maximum # events to write to disk

c*************************************************************************
c	Parameters used for parallel running
c*************************************************************************
      integer    max_host   ,maxplace    ,maxpoints    ,maxans
      parameter (max_host=9,maxplace=9,maxpoints=10,maxans=5)
c      parameter (max_host=99,maxplace=199,maxpoints=100,maxans=50)
      integer    maxprb
      parameter (maxprb = maxconfigs*maxplace*maxpoints)
      integer    maxfprb
      parameter (maxfprb = maxinvar*maxplace*maxpoints)

c*************************************************************************
c	Parameters for helicity sums in matrixN.f
c*************************************************************************
c      REAL*8 LIMHEL
c      PARAMETER(LIMHEL=1e-8) -> pass in the run_card.dat
      INTEGER MAXTRIES
      PARAMETER(MAXTRIES=25)
C     To pass the helicity configuration chosen by the DiscreteSampler to 
C     matrix<i>.f
      double precision hel_jacobian
	  INTEGER HEL_PICKED
	  COMMON/HEL_PICKED/HEL_PICKED,hel_jacobian
	  integer ee_picked
c same for MC for ee case	  
	  double precision ee_jacobian
	  COMMON/EE_PICKED/EE_PICKED,ee_jacobian
c**********************************************************************************
c       parameters to control is zero-cross section is due to PS or matrix-element
c**********************************************************************************
      integer nb_pass_cuts
      common/cut_efficiency/nb_pass_cuts
