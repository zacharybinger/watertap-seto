###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################

from copy import deepcopy

# Import Pyomo libraries
from pyomo.environ import (
    ConcreteModel,
    value,
    Set,
    Var,
    Param,
    Suffix,
    Constraint,
    check_optimal_termination,
    assert_optimal_termination,
    units as pyunits,
)

import pyomo.environ as pyo
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
)
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.exceptions import ConfigurationError, InitializationError
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    unscaled_variables_generator,
    unscaled_constraints_generator,
    badly_scaled_var_generator,
)

_log = idaeslog.getLogger(__name__)
__author__ = "Zachary Binger"


@declare_process_block_class("MDdummy")
class VAGMDData(UnitModelBlockData):
    """
    Vacuumed Membrane distillation (air-gapped) - batch operation model
    """

    CONFIG = ConfigBlock()

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be True",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False. """,
        ),
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be True",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. The filtration unit does not have defined volume, thus
    this must be False.""",
        ),
    )
    CONFIG.declare(
        "property_package_seawater",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_water",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )
    
    def build(self):
        super().build()

        """
        Model parameters
        """

        self.heat_exchanger_area = Param(
            initialize=1.34,
            units=pyunits.m**2,
            doc="Effective heat transfer coefficient",
        )

        self.STEC = Param(
            initialize=95,
            units=pyunits.kWh / pyunits.m**3,
            doc="Effective heat transfer coefficient",
        )

        self.membrane_area = Var(
                initialize=7.2, units=pyunits.m**2, doc="Area of module AS7C1.5L"
            )
        """
        Intermediate variables
        """        

        self.permeate_flux = Var(
            initialize=10,
            bounds=(0, None),
            units=pyunits.L / pyunits.h / pyunits.m**2,
            doc="Permeate flux",
        )

        self.permeate_flow_rate = Var(
            initialize=1,
            bounds=(0, None),
            units=pyunits.m**3 / pyunits.s,
            doc="Permeate flow rate",
        )

        """
        Output variables
        """
        self.thermal_power = Var(
            initialize=10,
            bounds=(0, None),
            units=pyunits.kW,
            doc="Thermal power requirment",
        )

        @self.Constraint(doc="Permeate flow rate")
        def eq_permeate_volumetric_flow_rate(b):
            return b.permeate_flow_rate == pyunits.convert(
                b.permeate_flux * self.membrane_area,
                to_units=pyunits.m**3 / pyunits.s,
            )

        @self.Constraint(doc="Thermal power")
        def eq_thermal_power(b): 
            return b.thermal_power == pyunits.convert((b.STEC * b.permeate_flow_rate), to_units=pyunits.kW)


    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        iscale.set_scaling_factor(self.permeate_flow_rate, 1e5)

        sf = iscale.get_scaling_factor(self.thermal_power)
        iscale.constraint_scaling_transform(self.eq_thermal_power, 1e-2)

if __name__=="__main__":
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.VAGMD = MDdummy()

    m.fs.VAGMD.STEC = 100
    m.fs.VAGMD.permeate_flux.fix(5)
    m.fs.VAGMD.membrane_area.fix(10)

    solver = get_solver()
    results = solver.solve(m, tee=False)

    print(f'{"Thermal Power: ":<24s}',  f'{value(m.fs.VAGMD.thermal_power):<5,.1f}', f'{pyunits.get_units(m.fs.VAGMD.thermal_power)}')
    print(f'{"Permeate Production: ":<24s}',  f'{value(pyunits.convert(m.fs.VAGMD.permeate_flow_rate, to_units=pyunits.L / pyunits.h)):<5,.1f}', f'{pyunits.get_units(pyunits.convert(m.fs.VAGMD.permeate_flow_rate, to_units=pyunits.L / pyunits.h))}')