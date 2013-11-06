using BinDeps

@BinDeps.setup

libnlopt = library_dependency("libnlopt", aliases=["libnlopt_cxx"])

provides(Sources,URI("http://ab-initio.mit.edu/nlopt/nlopt-2.4.tar.gz"), libnlopt)

provides(BuildProcess,Autotools(configure_options =
    ["--enable-shared", "--without-guile", "--without-python",
    "--without-octave", "--without-matlab","--with-cxx"],
    libtarget="libnlopt_cxx.la"),libnlopt, os = :Unix)

@osx_only begin
    using Homebrew
    provides( Homebrew.HB, "nlopt", libnlopt, os = :Darwin )
end

@BinDeps.install
