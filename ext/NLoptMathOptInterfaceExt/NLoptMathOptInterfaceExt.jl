# Copyright (c) 2013: Steven G. Johnson and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module NLoptMathOptInterfaceExt

import MathOptInterface as MOI
import NLopt

function __init__()
    # we need to add extension types back to the toplevel module
    @static if VERSION >= v"1.9"
        setglobal!(NLopt, :Optimizer, Optimizer)
    end
    return
end

include("MOI_wrapper.jl")
include("legacy.jl")

end  # module NLoptMathOptInterfaceExt
