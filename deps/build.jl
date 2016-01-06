using BinDeps

@BinDeps.setup

libnlopt = library_dependency("libnlopt", aliases=["libnlopt_cxx", "libnlopt$(WORD_SIZE)", "libnlopt-0"])

provides(AptGet, "libnlopt0", libnlopt)

provides(Sources,URI("http://ab-initio.mit.edu/nlopt/nlopt-2.4.tar.gz"), libnlopt)

provides(BuildProcess,Autotools(configure_options =
    ["--enable-shared", "--without-guile", "--without-python",
    "--without-octave", "--without-matlab","--with-cxx"],
    libtarget="libnlopt_cxx.la"),libnlopt, os = :Unix)

@osx_only begin
    using Homebrew
    provides( Homebrew.HB, "nlopt", libnlopt, os = :Darwin )
end

nloptname = "nlopt-2.4.1"

libdir = BinDeps.libdir(libnlopt)
srcdir = BinDeps.srcdir(libnlopt)
downloadsdir = BinDeps.downloadsdir(libnlopt)
extractdir(w) = joinpath(srcdir,"w$w")
destw(w) = joinpath(libdir,"libnlopt$(w).dll")

provides(Binaries, URI("http://ab-initio.mit.edu/nlopt/$(nloptname)-dll$(WORD_SIZE).zip"),
         libnlopt, unpacked_dir=".", os = :Windows)

@BinDeps.install Dict(:libnlopt => :libnlopt)
