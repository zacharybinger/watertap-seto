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
    m.fs.FPC = FlatPlatePhysical()
    m.fs.TES = HeatStorage()

    # Initial values for Flat Plate Collector
    m.fs.FPC.area_coll.set_value(2.98)
    m.fs.FPC.FRta.set_value(0.689)
    m.fs.FPC.FRUL.set_value(3.85)
    m.fs.FPC.iam.set_value(0.2)
    m.fs.FPC.mdot_test.set_value(0.045528)
    m.fs.FPC.cp_test.set_value(3400)  # specific heat of glycol [J/kg-K]
    m.fs.FPC.cp_use.set_value(3400)  # specific heat of glycol [J/kg-K]
    m.fs.FPC.ncoll.set_value(2)
    m.fs.FPC.pump_watts.set_value(45)
    m.fs.FPC.pump_eff.set_value(0.85)
    m.fs.FPC.T_amb.set_value(12)  # default SAM model at noon on Jan. 1
    m.fs.FPC.T_in.set_value(38.2)  # default SAM model at noon on Jan. 1
    m.fs.FPC.G_trans.set_value(540)  # default SAM model at noon on Jan. 1

    # Initial values for Membrane Distillation
    m.fs.VAGMD.STEC = 100
    m.fs.VAGMD.permeate_flow_rate.fix(5e-5)

    # initial values for Thermal Storage
    m.fs.TES.storage_eff.set_value(0.95)
    m.fs.TES.supply_eff.set_value(0.95)
    m.fs.TES.dt.set_value(1) # hr

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
        heat_gen == b.fs.fpc_to_md + b.fs.TES.heat_in[0] + b.fs.curtailment
        )

    @m.Constraint(doc="MD heat requirment")
    def eq_md_heat_req(b):
        return (md_heat_req == b.fs.fpc_to_md + b.fs.TES.heat_out[0] + b.fs.grid_to_md
        )

    # Add grid electricity cost
    @m.Expression(doc="grid cost")
    def grid_cost(b):
        return (heat_price * b.fs.grid_to_md * b.fs.TES.dt)
       
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
    m.fs.TES.initialize(outlvl=outlvl)
    m.fs.FPC.initialize(outlvl=outlvl)

    return 

if __name__ == "__main__":
    m = build_flat_plate_flowsheet()
    fix_dof_and_initialize(m)
    results = solver.solve(m)

    print(f'{"FPC Heat Gen: ":<24s}',  f'{value(m.fs.FPC.Q_useful):<10,.1f}', f'{pyunits.get_units(m.fs.FPC.Q_useful)}')
    

    print(f'{"initial state of heat: ":<24s}',  f'{value(m.fs.TES.initial_state_of_charge):<10,.1f}', f'{pyunits.get_units(m.fs.TES.initial_state_of_charge)}')
    print(f'{"state of heat: ":<24s}',          f'{value(m.fs.TES.state_of_charge[0]):<10,.1f}', f'{pyunits.get_units(m.fs.TES.state_of_charge[0])}')
    print(f'{"Heat throughput: ":<24s}',        f'{value(m.fs.TES.heat_throughput[0]):<10,.1f}', f'{pyunits.get_units(m.fs.TES.heat_throughput[0])}')
    print(f'{"TES power: ":<24s}',            f'{value(m.fs.TES.heat_output):<10,.1f}', f'{pyunits.get_units(m.fs.TES.heat_output)}')
    print(f'{"TES heat: ":<24s}',           f'{value(m.fs.TES.heat_capacity):<10,.1f}', f'{pyunits.get_units(m.fs.TES.heat_capacity)}')
    print(f'{"Heat generation: ":<24s}',   f'{value(m.fs.heat_generation):<10,.1f}', f'{pyunits.get_units(m.fs.heat_generation)}')