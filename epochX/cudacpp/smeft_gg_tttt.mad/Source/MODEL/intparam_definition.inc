ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

C     Parameters that should not be recomputed event by event.
C     
      IF(READLHA) THEN

        G = 2 * DSQRT(AS*PI)  ! for the first init

        MDL_COMPLEXI = DCMPLX(0.000000D+00,1.000000D+00)

        MDL_CUH = MDL_CUHRE+MDL_CUHIM*MDL_COMPLEXI

        MDL_CTHH = MDL_CTHRE+MDL_CTHIM*MDL_COMPLEXI

        MDL_CDH = MDL_CDHRE+MDL_CDHIM*MDL_COMPLEXI

        MDL_CBH = MDL_CBHRE+MDL_CBHIM*MDL_COMPLEXI

        MDL_CUG = MDL_CUGRE+MDL_CUGIM*MDL_COMPLEXI

        MDL_CTG = MDL_CTGRE+MDL_CTGIM*MDL_COMPLEXI

        MDL_CUW = MDL_CUWRE+MDL_CUWIM*MDL_COMPLEXI

        MDL_CTW = MDL_CTWRE+MDL_CTWIM*MDL_COMPLEXI

        MDL_CUB = MDL_CUBRE+MDL_CUBIM*MDL_COMPLEXI

        MDL_CTB = MDL_CTBRE+MDL_CTBIM*MDL_COMPLEXI

        MDL_CDG = MDL_CDGRE+MDL_CDGIM*MDL_COMPLEXI

        MDL_CBG = MDL_CBGRE+MDL_CBGIM*MDL_COMPLEXI

        MDL_CDW = MDL_CDWRE+MDL_CDWIM*MDL_COMPLEXI

        MDL_CBW = MDL_CBWRE+MDL_CBWIM*MDL_COMPLEXI

        MDL_CDB = MDL_CDBRE+MDL_CDBIM*MDL_COMPLEXI

        MDL_CBBB = MDL_CBBRE+MDL_CBBIM*MDL_COMPLEXI

        MDL_CHUD = MDL_CHUDRE+MDL_CHUDIM*MDL_COMPLEXI

        MDL_CHTB = MDL_CHTBRE+MDL_CHTBIM*MDL_COMPLEXI

        MDL_CUTBD1 = MDL_CUTBD1RE+MDL_CUTBD1IM*MDL_COMPLEXI

        MDL_CUTBD8 = MDL_CUTBD8RE+MDL_CUTBD8IM*MDL_COMPLEXI

        MDL_CJQTU1 = MDL_CJQTU1RE+MDL_CJQTU1IM*MDL_COMPLEXI

        MDL_CJQTU8 = MDL_CJQTU8RE+MDL_CJQTU8IM*MDL_COMPLEXI

        MDL_CJQBD1 = MDL_CJQBD1RE+MDL_CJQBD1IM*MDL_COMPLEXI

        MDL_CJQBD8 = MDL_CJQBD8RE+MDL_CJQBD8IM*MDL_COMPLEXI

        MDL_CJUJD1 = MDL_CJUJD1RE+MDL_CJUJD1IM*MDL_COMPLEXI

        MDL_CJUJD8 = MDL_CJUJD8RE+MDL_CJUJD8IM*MDL_COMPLEXI

        MDL_CJUJD11 = MDL_CJUJD11RE+MDL_CJUJD11IM*MDL_COMPLEXI

        MDL_CJUJD81 = MDL_CJUJD81RE+MDL_CJUJD81IM*MDL_COMPLEXI

        MDL_CQTJD1 = MDL_CQTJD1RE+MDL_CQTJD1IM*MDL_COMPLEXI

        MDL_CQTJD8 = MDL_CQTJD8RE+MDL_CQTJD8IM*MDL_COMPLEXI

        MDL_CJUQB1 = MDL_CJUQB1RE+MDL_CJUQB1IM*MDL_COMPLEXI

        MDL_CJUQB8 = MDL_CJUQB8RE+MDL_CJUQB8IM*MDL_COMPLEXI

        MDL_CQUJB1 = MDL_CQUJB1RE+MDL_CQUJB1IM*MDL_COMPLEXI

        MDL_CQUJB8 = MDL_CQUJB8RE+MDL_CQUJB8IM*MDL_COMPLEXI

        MDL_CJTQD1 = MDL_CJTQD1RE+MDL_CJTQD1IM*MDL_COMPLEXI

        MDL_CJTQD8 = MDL_CJTQD8RE+MDL_CJTQD8IM*MDL_COMPLEXI

        MDL_CQTQB1 = MDL_CQTQB1RE+MDL_CQTQB1IM*MDL_COMPLEXI

        MDL_CQTQB8 = MDL_CQTQB8RE+MDL_CQTQB8IM*MDL_COMPLEXI

        MDL_CEH = MDL_CEHRE+MDL_CEHIM*MDL_COMPLEXI

        MDL_CEW = MDL_CEWRE+MDL_CEWIM*MDL_COMPLEXI

        MDL_CEB = MDL_CEBRE+MDL_CEBIM*MDL_COMPLEXI

        MDL_CLEDJ = MDL_CLEDJRE+MDL_CLEDJIM*MDL_COMPLEXI

        MDL_CLEBQ = MDL_CLEBQRE+MDL_CLEBQIM*MDL_COMPLEXI

        MDL_CLEJU1 = MDL_CLEJU1RE+MDL_CLEJU1IM*MDL_COMPLEXI

        MDL_CLEJU3 = MDL_CLEJU3RE+MDL_CLEJU3IM*MDL_COMPLEXI

        MDL_CLEQT1 = MDL_CLEQT1RE+MDL_CLEQT1IM*MDL_COMPLEXI

        MDL_CLEQT3 = MDL_CLEQT3RE+MDL_CLEQT3IM*MDL_COMPLEXI

        MDL_MWSM = MDL_MW

        MDL_MW__EXP__2 = MDL_MW**2

        MDL_MZ__EXP__2 = MDL_MZ**2

        MDL_SQRT__2 = SQRT(DCMPLX(2.000000D+00))

        MDL_NB__2__EXP__0_25 = 2.000000D+00**2.500000D-01

        MDL_MH__EXP__2 = MDL_MH**2

        MDL_STH2 = 1.000000D+00-MDL_MW__EXP__2/MDL_MZ__EXP__2

        MDL_NB__10__EXP___M_40 = 1.000000D+01**-40

        MDL_PROPCORR = ABS(MDL_LINEARPROPCORRECTIONS)
     $   /(ABS(MDL_LINEARPROPCORRECTIONS)+MDL_NB__10__EXP___M_40)

        MDL_MZ1 = MDL_MZ

        MDL_MH1 = MDL_MH

        MDL_MT1 = MDL_MT

        MDL_WZ1 = MDL_WZ

        MDL_WW1 = MDL_WW

        MDL_WH1 = MDL_WH

        MDL_WT1 = MDL_WT

        MDL_CTH = SQRT(DCMPLX(1.000000D+00-MDL_STH2))

        MDL_MW1 = MDL_MWSM

        MDL_SQRT__STH2 = SQRT(DCMPLX(MDL_STH2))

        MDL_STH = MDL_SQRT__STH2

        MDL_LAMBDASMEFT__EXP__2 = MDL_LAMBDASMEFT**2

        MDL_CONJG__CBH = CONJG(DCMPLX(MDL_CBH))

        MDL_CONJG__CTHH = CONJG(DCMPLX(MDL_CTHH))

        MDL_MT__EXP__2 = MDL_MT**2

        MDL_MH__EXP__6 = MDL_MH**6

        MDL_MWSM__EXP__6 = MDL_MWSM**6

        MDL_MH__EXP__4 = MDL_MH**4

        MDL_MWSM__EXP__4 = MDL_MWSM**4

        MDL_MWSM__EXP__2 = MDL_MWSM**2

        MDL_MZ__EXP__4 = MDL_MZ**4

        MDL_MZ__EXP__6 = MDL_MZ**6

        MDL_CTH__EXP__2 = MDL_CTH**2

        MDL_STH__EXP__2 = MDL_STH**2

        MDL_MB__EXP__2 = MDL_MB**2

        MDL_MZ__EXP__3 = MDL_MZ**3

        MDL_STH__EXP__4 = MDL_STH**4

        MDL_STH__EXP__6 = MDL_STH**6

        MDL_STH__EXP__3 = MDL_STH**3

        MDL_STH__EXP__5 = MDL_STH**5

        MDL_PROPCORR__EXP__2 = MDL_PROPCORR**2

        MDL_PROPCORR__EXP__3 = MDL_PROPCORR**3

        MDL_PROPCORR__EXP__4 = MDL_PROPCORR**4

        MDL_CTH__EXP__3 = MDL_CTH**3

        MDL_AEW = (MDL_GF*MDL_MW__EXP__2*(1.000000D+00-MDL_MW__EXP__2
     $   /MDL_MZ__EXP__2)*MDL_SQRT__2)/PI

        MDL_SQRT__GF = SQRT(DCMPLX(MDL_GF))

        MDL_VEVHAT = 1.000000D+00/(MDL_NB__2__EXP__0_25*MDL_SQRT__GF)

        MDL_LAM = (MDL_GF*MDL_MH__EXP__2)/MDL_SQRT__2

        MDL_SQRT__AEW = SQRT(DCMPLX(MDL_AEW))

        MDL_EE = 2.000000D+00*MDL_SQRT__AEW*SQRT(DCMPLX(PI))

        MDL_YB = (MDL_YMB*MDL_SQRT__2)/MDL_VEVHAT

        MDL_YC = (MDL_YMC*MDL_SQRT__2)/MDL_VEVHAT

        MDL_YDO = (MDL_YMDO*MDL_SQRT__2)/MDL_VEVHAT

        MDL_YE = (MDL_YME*MDL_SQRT__2)/MDL_VEVHAT

        MDL_YM = (MDL_YMM*MDL_SQRT__2)/MDL_VEVHAT

        MDL_YS = (MDL_YMS*MDL_SQRT__2)/MDL_VEVHAT

        MDL_YT = (MDL_YMT*MDL_SQRT__2)/MDL_VEVHAT

        MDL_YTAU = (MDL_YMTAU*MDL_SQRT__2)/MDL_VEVHAT

        MDL_YUP = (MDL_YMUP*MDL_SQRT__2)/MDL_VEVHAT

        MDL_VEVHAT__EXP__2 = MDL_VEVHAT**2

        MDL_DGF = ((2.000000D+00*MDL_CHL3-MDL_CLL1)*MDL_VEVHAT__EXP__2)
     $   /MDL_LAMBDASMEFT__EXP__2

        MDL_DKH = ((MDL_CHBOX-MDL_CHDD/4.000000D+00)
     $   *MDL_VEVHAT__EXP__2)/MDL_LAMBDASMEFT__EXP__2

        MDL_VEVT = (1.000000D+00+MDL_DGF/2.000000D+00)*MDL_VEVHAT

        MDL_G1 = MDL_EE/MDL_CTH

        MDL_GW = MDL_EE/MDL_STH

        MDL_YB0 = (1.000000D+00-MDL_DGF/2.000000D+00)*MDL_YB
     $   +(MDL_VEVHAT__EXP__2*MDL_CONJG__CBH)/(2.000000D+00
     $   *MDL_LAMBDASMEFT__EXP__2)

        MDL_YT0 = (1.000000D+00-MDL_DGF/2.000000D+00)*MDL_YT
     $   +(MDL_VEVHAT__EXP__2*MDL_CONJG__CTHH)/(2.000000D+00
     $   *MDL_LAMBDASMEFT__EXP__2)

        MDL_EE__EXP__2 = MDL_EE**2

        MDL_GHAA = (MDL_EE__EXP__2*(-1.750000D+00+(4.000000D+00
     $   *(3.333333D-01+(7.000000D+00*MDL_MH__EXP__2)/(3.600000D+02
     $   *MDL_MT__EXP__2)))/3.000000D+00-(2.900000D+01*MDL_MH__EXP__6)
     $   /(1.680000D+04*MDL_MWSM__EXP__6)-(1.900000D+01*MDL_MH__EXP__4)
     $   /(1.680000D+03*MDL_MWSM__EXP__4)-(1.100000D+01*MDL_MH__EXP__2)
     $   /(1.200000D+02*MDL_MWSM__EXP__2)))/(8.000000D+00*PI**2)

        MDL_GHZA = (MDL_EE__EXP__2*(((4.583333D-01+(2.900000D+01
     $   *MDL_MH__EXP__6)/(1.008000D+05*MDL_MWSM__EXP__6)+(1.900000D
     $   +01*MDL_MH__EXP__4)/(1.008000D+04*MDL_MWSM__EXP__4)
     $   +(1.100000D+01*MDL_MH__EXP__2)/(7.200000D+02*MDL_MWSM__EXP__2)
     $   +(MDL_MH__EXP__4*MDL_MZ__EXP__2)/(2.100000D+03
     $   *MDL_MWSM__EXP__6)+(MDL_MH__EXP__2*MDL_MZ__EXP__2)/(2.800000D
     $   +02*MDL_MWSM__EXP__4)+(7.000000D+00*MDL_MZ__EXP__2)
     $   /(1.800000D+02*MDL_MWSM__EXP__2)+(6.700000D+01*MDL_MH__EXP__2
     $   *MDL_MZ__EXP__4)/(1.008000D+05*MDL_MWSM__EXP__6)+(5.300000D
     $   +01*MDL_MZ__EXP__4)/(1.008000D+04*MDL_MWSM__EXP__4)
     $   +(4.300000D+01*MDL_MZ__EXP__6)/(5.040000D+04*MDL_MWSM__EXP__6)
     $   -(3.100000D+01*MDL_CTH__EXP__2)/(2.400000D+01*MDL_STH__EXP__2)
     $   -(2.900000D+01*MDL_CTH__EXP__2*MDL_MH__EXP__6)/(2.016000D+04
     $   *MDL_MWSM__EXP__6*MDL_STH__EXP__2)-(1.900000D+01
     $   *MDL_CTH__EXP__2*MDL_MH__EXP__4)/(2.016000D+03
     $   *MDL_MWSM__EXP__4*MDL_STH__EXP__2)-(1.100000D+01
     $   *MDL_CTH__EXP__2*MDL_MH__EXP__2)/(1.440000D+02
     $   *MDL_MWSM__EXP__2*MDL_STH__EXP__2)-(MDL_CTH__EXP__2
     $   *MDL_MH__EXP__4*MDL_MZ__EXP__2)/(5.600000D+02
     $   *MDL_MWSM__EXP__6*MDL_STH__EXP__2)-(3.100000D+01
     $   *MDL_CTH__EXP__2*MDL_MH__EXP__2*MDL_MZ__EXP__2)/(2.520000D+03
     $   *MDL_MWSM__EXP__4*MDL_STH__EXP__2)-(MDL_CTH__EXP__2
     $   *MDL_MZ__EXP__2)/(9.000000D+00*MDL_MWSM__EXP__2
     $   *MDL_STH__EXP__2)-(4.300000D+01*MDL_CTH__EXP__2
     $   *MDL_MH__EXP__2*MDL_MZ__EXP__4)/(2.016000D+04
     $   *MDL_MWSM__EXP__6*MDL_STH__EXP__2)-(1.700000D+01
     $   *MDL_CTH__EXP__2*MDL_MZ__EXP__4)/(1.120000D+03
     $   *MDL_MWSM__EXP__4*MDL_STH__EXP__2)-(5.000000D+00
     $   *MDL_CTH__EXP__2*MDL_MZ__EXP__6)/(2.016000D+03
     $   *MDL_MWSM__EXP__6*MDL_STH__EXP__2))*MDL_STH)/MDL_CTH
     $   +((3.333333D-01+(7.000000D+00*MDL_MH__EXP__2)/(3.600000D+02
     $   *MDL_MT__EXP__2)+(1.100000D+01*MDL_MZ__EXP__2)/(3.600000D+02
     $   *MDL_MT__EXP__2))*(5.000000D-01-(4.000000D+00*MDL_STH__EXP__2)
     $   /3.000000D+00))/(MDL_CTH*MDL_STH)))/(4.000000D+00*PI**2)

        MDL_DMZ2 = ((MDL_CHDD/2.000000D+00+2.000000D+00*MDL_CHWB
     $   *MDL_CTH*MDL_STH)*MDL_VEVHAT__EXP__2)/MDL_LAMBDASMEFT__EXP__2

        MDL_DMH2 = 2.000000D+00*MDL_DKH-(3.000000D+00*MDL_CH
     $   *MDL_VEVHAT__EXP__2)/(2.000000D+00*MDL_LAM
     $   *MDL_LAMBDASMEFT__EXP__2)

        MDL_DGW = -MDL_DGF/2.000000D+00

        MDL_BARLAM = (1.000000D+00-MDL_DGF-MDL_DMH2)*MDL_LAM

        MDL_DWT = 2.000000D+00*MDL_WT*(MDL_DGW+(MDL_VEVHAT*(MDL_EE
     $   *(3.000000D+00*MDL_CHTBRE*MDL_MB*MDL_MT*MDL_MWSM__EXP__2
     $   +MDL_CHQ3*((MDL_MB__EXP__2-MDL_MT__EXP__2)**2+(MDL_MB__EXP__2
     $   +MDL_MT__EXP__2)*MDL_MWSM__EXP__2-2.000000D+00
     $   *MDL_MWSM__EXP__4))*MDL_VEVHAT+6.000000D+00*MDL_MWSM__EXP__2
     $   *(MDL_CTWRE*MDL_MT*(MDL_MB__EXP__2-MDL_MT__EXP__2
     $   +MDL_MWSM__EXP__2)+MDL_CBWRE*MDL_MB*(-MDL_MB__EXP__2
     $   +MDL_MT__EXP__2+MDL_MWSM__EXP__2))*MDL_STH*MDL_SQRT__2))
     $   /(MDL_EE*MDL_LAMBDASMEFT__EXP__2*((MDL_MB__EXP__2
     $   -MDL_MT__EXP__2)**2+(MDL_MB__EXP__2+MDL_MT__EXP__2)
     $   *MDL_MWSM__EXP__2-2.000000D+00*MDL_MWSM__EXP__4)))

        MDL_DWW = (2.000000D+00*MDL_DGW+(2.000000D+00*(2.000000D+00
     $   *MDL_CHJ3+MDL_CHL3)*MDL_VEVHAT__EXP__2)/(3.000000D+00
     $   *MDL_LAMBDASMEFT__EXP__2))*MDL_WW

        MDL_GWSH = (MDL_EE*(1.000000D+00+MDL_DGW-(MDL_CHW
     $   *MDL_VEVHAT__EXP__2)/MDL_LAMBDASMEFT__EXP__2))/MDL_STH

        MDL_VEV = (1.000000D+00-(3.000000D+00*MDL_CH
     $   *MDL_VEVHAT__EXP__2)/(8.000000D+00*MDL_LAM
     $   *MDL_LAMBDASMEFT__EXP__2))*MDL_VEVT

        MDL_DG1 = (-MDL_DGF-MDL_DMZ2/MDL_STH__EXP__2)/2.000000D+00

        MDL_DWHC = MDL_YC/(MDL_YC+MDL_NB__10__EXP___M_40)*(-2.884000D
     $   -02*MDL_DGF+((5.768000D-02*MDL_CHBOX-1.442000D-02*MDL_CHDD
     $   -5.768000D-02*MDL_CUHRE)*MDL_VEVHAT__EXP__2)
     $   /MDL_LAMBDASMEFT__EXP__2)

        MDL_DWHB = MDL_YB/(MDL_YB+MDL_NB__10__EXP___M_40)
     $   *(MDL_VEVHAT__EXP__2*(-1.161800D+00*MDL_CBHRE)
     $   /(MDL_LAMBDASMEFT__EXP__2*(MDL_YB+MDL_NB__10__EXP___M_40))
     $   -5.809000D-01*MDL_DGF+(MDL_VEVHAT__EXP__2*(1.161800D+00
     $   *MDL_CHBOX-2.904500D-01*MDL_CHDD))/(MDL_LAMBDASMEFT__EXP__2))

        MDL_DWHTA = MDL_YTAU/(MDL_YTAU+MDL_NB__10__EXP___M_40)*(
     $   -6.256000D-02*MDL_DGF+MDL_VEVHAT__EXP__2*(-1.251200D-01
     $   *MDL_CEHRE+1.251200D-01*MDL_CHBOX-3.128000D-02*MDL_CHDD)
     $   /(MDL_LAMBDASMEFT__EXP__2))

        MDL_DWZ = MDL_WZ*(-1.000000D+00+(3.600000D+01*MDL_CTH*MDL_MB
     $   *MDL_MZ__EXP__2*MDL_STH*(MDL_CBWRE*MDL_CTH+MDL_CBBRE*MDL_STH)
     $   *(-3.000000D+00+4.000000D+00*MDL_STH__EXP__2)*MDL_VEVHAT
     $   *MDL_SQRT__2*SQRT(DCMPLX(-4.000000D+00*MDL_MB__EXP__2
     $   +MDL_MZ__EXP__2))+MDL_EE*MDL_LAMBDASMEFT__EXP__2*(2.000000D
     $   +00*MDL_MZ__EXP__3*(2.700000D+01+5.400000D+01*MDL_DGW
     $   -5.400000D+01*(1.000000D+00+MDL_DG1+MDL_DGW)*MDL_STH__EXP__2
     $   +7.600000D+01*(1.000000D+00+4.000000D+00*MDL_DG1-2.000000D+00
     $   *MDL_DGW)*MDL_STH__EXP__4+1.520000D+02*(-MDL_DG1+MDL_DGW)
     $   *MDL_STH__EXP__6)+MDL_MZ__EXP__2*(9.000000D+00+1.800000D+01
     $   *MDL_DGW-6.000000D+00*(2.000000D+00+MDL_DG1+3.000000D+00
     $   *MDL_DGW)*MDL_STH__EXP__2+8.000000D+00*(1.000000D+00
     $   +4.000000D+00*MDL_DG1-2.000000D+00*MDL_DGW)*MDL_STH__EXP__4
     $   +1.600000D+01*(-MDL_DG1+MDL_DGW)*MDL_STH__EXP__6)
     $   *SQRT(DCMPLX(-4.000000D+00*MDL_MB__EXP__2+MDL_MZ__EXP__2))
     $   +MDL_MB__EXP__2*(-9.000000D+00-1.800000D+01*MDL_DGW-6.000000D
     $   +00*(4.000000D+00+1.100000D+01*MDL_DG1-3.000000D+00*MDL_DGW)
     $   *MDL_STH__EXP__2+1.600000D+01*(1.000000D+00+4.000000D+00
     $   *MDL_DG1-2.000000D+00*MDL_DGW)*MDL_STH__EXP__4+3.200000D+01*(
     $   -MDL_DG1+MDL_DGW)*MDL_STH__EXP__6)*SQRT(DCMPLX(-4.000000D+00
     $   *MDL_MB__EXP__2+MDL_MZ__EXP__2)))+2.000000D+00*MDL_EE
     $   *MDL_VEVHAT__EXP__2*(3.600000D+01*MDL_CHJ3*MDL_MZ__EXP__3
     $   +1.800000D+01*MDL_CHL3*MDL_MZ__EXP__3+9.000000D+00*(3.000000D
     $   +00*MDL_CHBQ-MDL_CHQ1-MDL_CHQ3)*MDL_MB__EXP__2*SQRT(DCMPLX(
     $   -4.000000D+00*MDL_MB__EXP__2+MDL_MZ__EXP__2))+9.000000D+00
     $   *MDL_CHQ1*MDL_MZ__EXP__2*SQRT(DCMPLX(-4.000000D+00
     $   *MDL_MB__EXP__2+MDL_MZ__EXP__2))+9.000000D+00*MDL_CHQ3
     $   *MDL_MZ__EXP__2*SQRT(DCMPLX(-4.000000D+00*MDL_MB__EXP__2
     $   +MDL_MZ__EXP__2))+3.000000D+00*MDL_CHWB*MDL_CTH*(-7.000000D
     $   +00*MDL_MB__EXP__2+MDL_MZ__EXP__2)*MDL_STH*SQRT(DCMPLX(
     $   -4.000000D+00*MDL_MB__EXP__2+MDL_MZ__EXP__2))+8.000000D+00
     $   *MDL_CHWB*MDL_CTH*MDL_STH__EXP__3*(2.000000D+00
     $   *MDL_MB__EXP__2*SQRT(DCMPLX(-4.000000D+00*MDL_MB__EXP__2
     $   +MDL_MZ__EXP__2))+MDL_MZ__EXP__2*(1.900000D+01*MDL_MZ
     $   +SQRT(DCMPLX(-4.000000D+00*MDL_MB__EXP__2+MDL_MZ__EXP__2))))
     $   -8.000000D+00*MDL_CHWB*MDL_CTH*MDL_STH__EXP__5*(2.000000D+00
     $   *MDL_MB__EXP__2*SQRT(DCMPLX(-4.000000D+00*MDL_MB__EXP__2
     $   +MDL_MZ__EXP__2))+MDL_MZ__EXP__2*(1.900000D+01*MDL_MZ
     $   +SQRT(DCMPLX(-4.000000D+00*MDL_MB__EXP__2+MDL_MZ__EXP__2))))
     $   -6.000000D+00*MDL_STH__EXP__2*(2.000000D+00*(MDL_CHBQ
     $   +MDL_CHQ1+MDL_CHQ3)*MDL_MB__EXP__2*SQRT(DCMPLX(-4.000000D+00
     $   *MDL_MB__EXP__2+MDL_MZ__EXP__2))+MDL_MZ__EXP__2*((2.000000D
     $   +00*MDL_CHD+3.000000D+00*MDL_CHE-2.000000D+00*MDL_CHJ1
     $   +3.000000D+00*(2.000000D+00*MDL_CHJ3+MDL_CHL1+MDL_CHL3)
     $   -4.000000D+00*MDL_CHU)*MDL_MZ+(MDL_CHBQ+MDL_CHQ1+MDL_CHQ3)
     $   *SQRT(DCMPLX(-4.000000D+00*MDL_MB__EXP__2+MDL_MZ__EXP__2))))))
     $   /(MDL_EE*MDL_LAMBDASMEFT__EXP__2*(2.000000D+00*MDL_MZ__EXP__3
     $   *(2.700000D+01-5.400000D+01*MDL_STH__EXP__2+7.600000D+01
     $   *MDL_STH__EXP__4)+MDL_MZ__EXP__2*(9.000000D+00-1.200000D+01
     $   *MDL_STH__EXP__2+8.000000D+00*MDL_STH__EXP__4)*SQRT(DCMPLX(
     $   -4.000000D+00*MDL_MB__EXP__2+MDL_MZ__EXP__2))+MDL_MB__EXP__2
     $   *(-9.000000D+00-2.400000D+01*MDL_STH__EXP__2+1.600000D+01
     $   *MDL_STH__EXP__4)*SQRT(DCMPLX(-4.000000D+00*MDL_MB__EXP__2
     $   +MDL_MZ__EXP__2)))))

        MDL_G1SH = (MDL_EE*(1.000000D+00+MDL_DG1-(MDL_CHB
     $   *MDL_VEVHAT__EXP__2)/MDL_LAMBDASMEFT__EXP__2))/MDL_CTH

        MDL_EE__EXP__3 = MDL_EE**3

        MDL_VEVHAT__EXP__3 = MDL_VEVHAT**3

      ENDIF
C     
C     Parameters that should be recomputed at an event by even basis.
C     
      AS = G**2/4/PI

      MDL_SQRT__AS = SQRT(DCMPLX(AS))

      MDL_GHGG2 = (-7.000000D+00*AS)/(7.200000D+02*PI)

      MDL_GHGG4 = AS/(3.600000D+02*PI)

      MDL_GHGG5 = AS/(2.000000D+01*PI)

      MDL_G__EXP__2 = G**2

      MDL_GHGG1 = MDL_G__EXP__2/(4.800000D+01*PI**2)

      MDL_GHGG3 = (AS*G)/(6.000000D+01*PI)

      MDL_G__EXP__3 = G**3

      MDL_DWH = MDL_WH*(-2.416100D-01*MDL_DGF+9.664400D-01*MDL_DGW
     $ +4.832200D-01*MDL_DKH-1.118651D-01*MDL_DWW+(3.641038D-01
     $ *MDL_CHJ3*MDL_VEVHAT__EXP__2)/MDL_LAMBDASMEFT__EXP__2
     $ +(1.760831D-01*MDL_CHL3*MDL_VEVHAT__EXP__2)
     $ /MDL_LAMBDASMEFT__EXP__2+(1.636000D-01*MDL_CHG*MDL_MT__EXP__2
     $ *MDL_VEVHAT__EXP__2)/(MDL_LAMBDASMEFT__EXP__2*(-5.000000D-01
     $ *MDL_GHGG2*MDL_MH__EXP__2+MDL_GHGG1*MDL_MT__EXP__2))+(MDL_CHW*(
     $ -3.593779D-01*MDL_GHAA*MDL_GHZA+6.164000D-03*MDL_CTH*MDL_GHAA
     $ *MDL_STH+4.540000D-03*MDL_GHZA*MDL_STH__EXP__2)
     $ *MDL_VEVHAT__EXP__2)/(MDL_GHAA*MDL_GHZA*MDL_LAMBDASMEFT__EXP__2)
     $ +(MDL_CHWB*(-4.540000D-03*MDL_CTH*MDL_GHZA*MDL_STH+MDL_GHAA*(
     $ -3.082000D-03+6.164000D-03*MDL_STH__EXP__2))*MDL_VEVHAT__EXP__2)
     $ /(MDL_GHAA*MDL_GHZA*MDL_LAMBDASMEFT__EXP__2)+(MDL_CHB*(
     $ -6.164000D-03*MDL_CTH*MDL_GHAA*MDL_STH-4.540000D-03*MDL_GHZA*(
     $ -1.000000D+00+MDL_STH__EXP__2))*MDL_VEVHAT__EXP__2)/(MDL_GHAA
     $ *MDL_GHZA*MDL_LAMBDASMEFT__EXP__2)+MDL_DWHC+MDL_DWHB+MDL_DWHTA)

C     
C     Parameters that should be updated for the loops.
C     
C     
C     Definition of the EW coupling used in the write out of aqed
C     
      GAL(1) = 2.378414230005442133435D0*MDL_MW*DSQRT(1D0-MDL_MW**2
     $ /MDL_MZ**2)*DSQRT(MDL_GF)
      GAL(2) = 1D0

