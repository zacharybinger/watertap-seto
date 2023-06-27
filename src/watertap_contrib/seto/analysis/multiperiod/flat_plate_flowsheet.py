# Pyomo imports
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Param,
    Expression,
    Constraint,
    Block,
    log10,
    TransformationFactory,
    assert_optimal_termination,
    value,
    units as pyunits,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.network import Port
# IDAES imports
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale
from idaes.core import FlowsheetBlock
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    unscaled_variables_generator,
    badly_scaled_var_generator,
)

# WaterTAP imports
from watertap_contrib.seto.core import PySAMWaterTAP
from idaes.models.unit_models import Product, Feed
# from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
# from watertap.property_models.water_prop_pack import WaterParameterBlock
# from watertap_contrib.seto.unit_models.surrogate import VAGMDSurrogate
from watertap_contrib.seto.solar_models.physical.flat_plate import FlatPlatePhysical
from watertap_contrib.seto.analysis.multiperiod.MD_dummy import MDdummy
from watertap_contrib.seto.analysis.multiperiod.storage_dummy import HeatStorage
from watertap.property_models.NaCl_prop_pack import NaClParameterBlock

_log = idaeslog.getLogger(__name__)
solver = get_solver()

def build_flat_plate_flowsheet(
    m=None,
    heat_price = 0.1,
    heat_gen = 3,
    md_heat_req = 18,
    md_capacity = 5
):
    """
    This function builds a unit model for a certain time period

    Returns:
        object: A Pyomo concrete optimization model and flowsheet
    """
    if m is None:
        m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = NaClParameterBlock()
    m.fs.VAGMD = MDdummy()

    storage = add_stroage(m)
    # m.fs.membrane_area = ro_elec_req
    m.fs.VAGMD.STEC = 100
    m.fs.VAGMD.permeate_flow_rate.fix(5e-5)

    if "USD_2021" not in pyunits._pint_registry:
            pyunits.load_definitions_from_strings(
                ["USD_2021 = 500/708.0 * USD_CE500"]
            )

    m.fs.fpc_to_md = Var(
            initialize = 100,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'PV to RO electricity'
        )
    m.fs.grid_to_md = Var(
            initialize = 100,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'Grid to RO electricity'
        )
    
    m.fs.curtailment = Var(
            initialize = 0,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'Heat curtailment'
        )
    
    m.fs.heat_price = Var(
            initialize = heat_price,
            bounds = (0,None),
            units = pyunits.USD_2021,
            doc = 'Electric Cost'
        )
    
    m.fs.heat_generation = Var(
            initialize = heat_gen,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'FPC heat Gen'
        )
    
    # Add heat flow balance
    @m.Constraint(doc="System heat flow")
    def eq_fpc_heat_gen(b):
        return (
        heat_gen == b.fs.fpc_to_md + b.fs.tes.heat_in[0] + b.fs.curtailment
        )

    @m.Constraint(doc="MD heat requirment")
    def eq_md_heat_req(b):
        return (md_heat_req == b.fs.fpc_to_md + b.fs.tes.heat_out[0] + b.fs.grid_to_md
        )

    # Add grid electricity cost
    @m.Expression(doc="grid cost")
    def grid_cost(b):
        return (heat_price * b.fs.grid_to_md * b.fs.tes.dt)
       
    # Add grid electricity cost
    @m.Constraint(doc="FPC heat generation")
    def fpc_heat_gen(b):
        return (heat_gen == m.fs.heat_generation * 1)


    return m

def fix_dof_and_initialize(
    m,
    outlvl=idaeslog.WARNING,
):
    """Fix degrees of freedom and initialize the flowsheet

    This function fixes the degrees of freedom of each unit and initializes the entire flowsheet.

    Args:
        m: Pyomo `Block` or `ConcreteModel` containing the flowsheet
        outlvl: Logger (default: idaeslog.WARNING)
    """
    m.fs.tes.initialize(outlvl=outlvl)

    return 

def add_stroage(m):
    m.fs.tes = HeatStorage()
    m.fs.tes.storage_eff.set_value(0.95)
    m.fs.tes.supply_eff.set_value(0.95)
    m.fs.tes.dt.set_value(1) # hr

    return m.fs.tes

if __name__ == "__main__":
    m = build_flat_plate_flowsheet()

    print(f'{value(m.fs.heat_generation):<10,.1f}', pyunits.get_units(m.fs.heat_generation))
    fix_dof_and_initialize(m)
    print(f'{value(m.fs.heat_generation):<10,.1f}', pyunits.get_units(m.fs.heat_generation))
    results = solver.solve(m)
    print(f'{value(m.fs.heat_generation):<10,.1f}', pyunits.get_units(m.fs.heat_generation))

    print(f'{"initial state of heat: ":<24s}',  f'{value(m.fs.tes.initial_state_of_charge):<10,.1f}', f'{pyunits.get_units(m.fs.tes.initial_state_of_charge)}')
    print(f'{"state of heat: ":<24s}',          f'{value(m.fs.tes.state_of_charge[0]):<10,.1f}', f'{pyunits.get_units(m.fs.tes.state_of_charge[0])}')
    print(f'{"Heat throughput: ":<24s}',        f'{value(m.fs.tes.heat_throughput[0]):<10,.1f}', f'{pyunits.get_units(m.fs.tes.heat_throughput[0])}')
    print(f'{"TES power: ":<24s}',            f'{value(m.fs.tes.heat_output):<10,.1f}', f'{pyunits.get_units(m.fs.tes.heat_output)}')
    print(f'{"TES heat: ":<24s}',           f'{value(m.fs.tes.heat_capacity):<10,.1f}', f'{pyunits.get_units(m.fs.tes.heat_capacity)}')
    print(f'{"Heat generation: ":<24s}',   f'{value(m.fs.heat_generation):<10,.1f}', f'{pyunits.get_units(m.fs.heat_generation)}')