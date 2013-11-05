using BinDeps

@BinDeps.setup

libnlopt = library_dependency("libnlopt", aliases=["libnlopt_cxx"])

provides(Sources,URI("http://ab-initio.mit.edu/nlopt/nlopt-2.4.tar.gz"), libnlopt)

provides(BuildProcess,Autotools(configure_options =
    ["--enable-shared", "--without-guile", "--without-python",
    "--without-octave", "--without-matlab","--with-cxx"],
    libtarget="libnlopt_cxx.la"),libnlopt, os = :Unix)

@BinDeps.install
