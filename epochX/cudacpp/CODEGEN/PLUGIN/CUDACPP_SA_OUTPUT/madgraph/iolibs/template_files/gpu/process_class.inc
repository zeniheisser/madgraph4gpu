! Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
! Created by: J. Alwall (Jul 2010) for the MG5aMC CPP backend.
!==========================================================================
! Copyright (C) 2020-2024 CERN and UCLouvain.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Modified by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
! Further modified by: A. Valassi (2021-2024) for the MG5aMC CUDACPP plugin.
!==========================================================================
  //==========================================================================
  // A class for calculating the matrix elements for
  %(process_lines)s
  //--------------------------------------------------------------------------

  class CPPProcess
  {
  public: /* clang-format off */

    // Constructor (from command line arguments)
    CPPProcess( bool verbose = false, bool debug = false );

    // Destructor
    ~CPPProcess();

    // Initialize process (read model parameters from file)
    virtual void initProc( const std::string& param_card_name );

    // Retrieve the compiler that was used to build this module
    static const std::string getCompiler();

    // Other methods of this instance (???)
    //const std::vector<fptype>& getMasses() const { return m_masses; }
    //virtual int code() const{ return 1; }
    //void setInitial( int inid1, int inid2 ){ id1 = inid1; id2 = inid2; }
    //int getDim() const { return dim; }
    //int getNIOParticles() const { return nexternal; } // nexternal was nioparticles

    // Accessors (unused so far: add four of them only to fix a clang build warning)
    //bool verbose() const { return m_verbose; }
    bool debug() const { return m_debug; }

  public:

    // Process-independent compile-time constants
    static constexpr int np4 = 4; // dimensions of 4-momenta (E,px,py,pz)
    static constexpr int nw6 = 6; // dimensions of each wavefunction (HELAS KEK 91-11): e.g. 6 for e+ e- -> mu+ mu- (fermions and vectors)

    // Process-dependent compile-time constants
    static constexpr int npari = %(nincoming)d; // #particles in the initial state (incoming): e.g. 2 (e+ e-) for e+ e- -> mu+ mu-
    static constexpr int nparf = %(noutcoming)d; // #particles in the final state (outgoing): e.g. 2 (mu+ mu-) for e+ e- -> mu+ mu-
    static constexpr int npar = npari + nparf; // #particles in total (external = initial + final): e.g. 4 for e+ e- -> mu+ mu-
    static constexpr int ncomb = %(nbhel)d; // #helicity combinations: e.g. 16 for e+ e- -> mu+ mu- (2**4 = fermion spin up/down ** npar)
    static constexpr int ndiagrams = %(ndiagrams)d; // #Feynman diagrams: e.g. 3 for e+ e- -> mu+ mu-

    // Hardcoded parameters for this process (constant class variables)
    // [NB: this class assumes nprocesses==1 i.e. a single DSIG1 and no DSIG2 in Fortran (#272 and #343)]
    // [NB: these parameters (e.g. nwf) are P1-specific, i.e. they are different for different P1 subdirectories (#644)]
    // [NB: I am currently unable to get the right value of nwf in CPPProcess.h - will hardcode it in CPPProcess.cc instead (#644)]
    //static const int nwf = ??; // #wavefunctions = #external (npar) + #internal: e.g. 5 for e+ e- -> mu+ mu- (1 internal is gamma or Z)

    // Other variables of this instance (???)
    //static const int ninitial = CPPProcess::npari;
    //static const int nexternal = %(nexternal)d; // CPPProcess::npar (nexternal was nioparticles)
    //static const int nwavefuncs = %(nwavefuncs)d; // (?!?! this should be nwf but export_cpp gives the wrong value here)
    //static const int namplitudes = %(namp)d;
    //static const int ncomb = %(ncomb)d; // CPPProcess::ncomb

  private: /* clang-format on */

    // Command line arguments (constructor)
    bool m_verbose;
    bool m_debug;

    // Physics model parameters to be read from file (initProc function)
#ifndef MGONGPU_HARDCODE_PARAM
    Parameters_%(model_name)s* m_pars;
#endif
    std::vector<fptype> m_masses; // external particle masses

    // Other variables of this instance (???)
    //int id1, id2; // initial particle ids
    //cxtype** amp; // ???
  };
