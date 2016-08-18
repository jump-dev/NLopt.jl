using BinDeps
using Compat

@BinDeps.setup

libnlopt = library_dependency("libnlopt", aliases=["libnlopt_cxx", "libnlopt$(Sys.WORD_SIZE)"])

provides(AptGet, "libnlopt0", libnlopt)

provides(Sources,URI("http://ab-initio.mit.edu/nlopt/nlopt-2.4.tar.gz"), libnlopt)

provides(BuildProcess,Autotools(configure_options =
    ["--enable-shared", "--without-guile", "--without-python",
    "--without-octave", "--without-matlab","--with-cxx"],
    libtarget="libnlopt_cxx.la"),libnlopt, os = :Unix)

if is_apple()
    using Homebrew
    provides( Homebrew.HB, "nlopt", libnlopt, os = :Darwin )
end

nloptname = "nlopt-2.4.1"

libdir = BinDeps.libdir(libnlopt)
srcdir = BinDeps.srcdir(libnlopt)
downloadsdir = BinDeps.downloadsdir(libnlopt)
extractdir(w) = joinpath(srcdir,"w$w")
destw(w) = joinpath(libdir,"libnlopt$(w).dll")

type FileCopyRule <: BinDeps.BuildStep
    src::AbstractString
    dest::AbstractString
end
Base.run(fc::FileCopyRule) = isfile(fc.dest) || cp(fc.src, fc.dest)

provides(BuildProcess,
	(@build_steps begin
		FileDownloader("http://ab-initio.mit.edu/nlopt/$(nloptname)-dll32.zip", joinpath(downloadsdir, "$(nloptname)-dll32.zip"))
		FileDownloader("http://ab-initio.mit.edu/nlopt/$(nloptname)-dll64.zip", joinpath(downloadsdir, "$(nloptname)-dll64.zip"))
		CreateDirectory(srcdir, true)
		CreateDirectory(joinpath(srcdir,"w32"), true)
		CreateDirectory(joinpath(srcdir,"w64"), true)
		FileUnpacker(joinpath(downloadsdir,"$(nloptname)-dll32.zip"), extractdir(32), joinpath(extractdir(32),"matlab"))
		FileUnpacker(joinpath(downloadsdir,"$(nloptname)-dll64.zip"), extractdir(64), joinpath(extractdir(64),"matlab"))
		CreateDirectory(libdir, true)
        FileCopyRule(joinpath(extractdir(32),"libnlopt-0.dll"), destw(32))
        FileCopyRule(joinpath(extractdir(64),"libnlopt-0.dll"), destw(64))
	end), libnlopt, os = :Windows)

if is_windows()
    push!(BinDeps.defaults, BuildProcess)
end

@BinDeps.install Dict(:libnlopt => :libnlopt)

if is_windows()
    pop!(BinDeps.defaults)
end
