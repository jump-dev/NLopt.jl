# Copyright (c) 2019 Mathieu Besançon, Oscar Dowson, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Clang.Generators
import NLopt_jll

c_api = joinpath(NLopt_jll.artifact_dir, "include", "nlopt.h")

build!(
    create_context(
        [c_api],
        get_default_args(),
        load_options(joinpath(@__DIR__, "generate.toml")),
    ),
)

header = """
# Copyright (c) 2013: Steven G. Johnson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

#! format: off

"""

filename = joinpath(@__DIR__, "..", "src", "libnlopt.jl")
contents = read(filename, String)
contents = replace(
    contents,
    "const nlopt_opt = Ptr{nlopt_opt_s}" => "const nlopt_opt = Ptr{Cvoid}",
)
write(filename, header * contents)
