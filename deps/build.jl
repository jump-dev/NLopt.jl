using BinDeps

@BinDeps.setup

libnlopt = library_dependency("libnlopt", aliases=["libnlopt_cxx", "libnlopt$(WORD_SIZE)"])

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
		@build_steps begin
			ChangeDirectory(extractdir(32))
			FileRule(destw(32), @build_steps begin
				`cp libnlopt-0.dll $(destw(32))`
				end)
		end
		@build_steps begin
			ChangeDirectory(extractdir(64))
			FileRule(destw(64), @build_steps begin
				`cp libnlopt-0.dll $(destw(64))`
				end)
		end
	end), libnlopt, os = :Windows)

@windows_only push!(BinDeps.defaults, BuildProcess)

@BinDeps.install

@windows_only pop!(BinDeps.defaults)
