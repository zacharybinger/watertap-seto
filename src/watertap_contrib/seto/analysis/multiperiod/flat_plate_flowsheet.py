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
from watertap_contrib.seto.solar_models.physical.flat_plate import IrradianceModelIsoSky
from watertap.property_models.NaCl_prop_pack import NaClParameterBlock

_log = idaeslog.getLogger(__name__)
solver = get_solver()

def build_flat_plate_flowsheet(
    m=None,
    heat_price = 0.1,
    # heat_gen = 3,
    md_heat_req = 18,
    md_capacity = 5,
    day = 51,
    hour = 9
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
    m.fs.isosky = IrradianceModelIsoSky()
    m.fs.FPC = FlatPlatePhysical()
    m.fs.TES = HeatStorage()

    print(' ============= Initializing Irradiance Model =============')
    initialize_isosky(m, day, hour)
    print(' ============= Initializing Flat Plate Collector =============')
    initialize_fpc(m)
    print(' ============= Initializing Thermal Energy Storage =============')
    initialize_tes(m)
    print(' ============= Initializing Membrane Distillation =============')
    initialize_vagmd(m)

    if "USD_2021" not in pyunits._pint_registry:
            pyunits.load_definitions_from_strings(
                ["USD_2021 = 500/708.0 * USD_CE500"]
            )

    m.fs.day = Var(
        initialize = day,
        bounds = (0,None),
        units = pyunits.day,
        doc = 'Day of the year'
    )
    
    m.fs.hour = Var(
        initialize = hour,
        bounds = (0,None),
        units = pyunits.hour,
        doc = 'Hour of the day'
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
            initialize = 1,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'FPC heat Gen'
        )
    
    
    # Add grid electricity cost
    @m.Constraint(doc="FPC heat generation")
    def fpc_heat_gen(b):
        return m.fs.heat_generation == m.fs.FPC.Q_useful
    
    # Add heat flow balance
    @m.Constraint(doc="System heat flow")
    def eq_fpc_heat_gen(b):
        return (
        m.fs.heat_generation == b.fs.fpc_to_md + b.fs.TES.heat_in[0] + b.fs.curtailment
        # heat_gen == b.fs.fpc_to_md + b.fs.TES.heat_in[0] + b.fs.curtailment
        )

    @m.Constraint(doc="MD heat requirment")
    def eq_md_heat_req(b):
        return (md_heat_req == b.fs.fpc_to_md + b.fs.TES.heat_out[0] + b.fs.grid_to_md
        )

    # Add grid electricity cost
    @m.Expression(doc="grid cost")
    def grid_cost(b):
        return (heat_price * b.fs.grid_to_md * b.fs.TES.dt)
       
    # # Add grid electricity cost
    # @m.Constraint(doc="FPC heat generation")
    # def eq_fpc_heat_gen(b):
    #     return (heat_gen == m.fs.FPC.Q_useful)
    #     # return (heat_gen == m.fs.heat_generation * 1)

    # # Add grid electricity cost
    # @m.Constraint(doc="FPC heat generation")
    # def fpc_heat_gen(b):
    #     return (heat_gen == m.fs.heat_generation * 1)
    
    return m

def initialize_isosky(m, day, hour):
    m.fs.isosky.phi.set_value(40)  # Latitude of collector
    m.fs.isosky.lon.set_value(89.4)  # Longitude of collector
    m.fs.isosky.std_meridian.set_value(90)  # Standard meridian corresponding to longitude
    m.fs.isosky.standard_time.set_value(hour)  # Standard time, 9:41 AM
    m.fs.isosky.beta.set_value(60)  # Tilt angle of collector
    m.fs.isosky.rho_g.set_value(0.6)  # Ground reflectance
    m.fs.isosky.day_of_year.set_value(day)  # Day of year (Feb 20th)
    m.fs.isosky.G_bn.set_value(305.40)  # Beam normal radiation
    m.fs.isosky.G_d.set_value(796)  # Diffuse radiation on horizontal surface

    m.fs.isosky.initialize(outlvl=idaeslog.WARNING)
    solver.solve(m.fs.isosky)
    print(f'{"ISO solar_time: ":<24s}',  f'{value(m.fs.isosky.solar_time):<10,.1f}', f'{pyunits.get_units(m.fs.isosky.solar_time)}')
    print(f'{"ISO omega: ":<24s}',  f'{value(m.fs.isosky.omega):<10,.1f}', f'{pyunits.get_units(m.fs.isosky.omega)}')
    print(f'{"ISO theta_z: ":<24s}',  f'{value(m.fs.isosky.theta_z):<10,.1f}', f'{pyunits.get_units(m.fs.isosky.theta_z)}')
    print(f'{"ISO R_b: ":<24s}',  f'{value(m.fs.isosky.R_b):<10,.1f}', f'{pyunits.get_units(m.fs.isosky.R_b)}')
    print(f'{"ISO theta: ":<24s}',  f'{value(m.fs.isosky.theta):<10,.1f}', f'{pyunits.get_units(m.fs.isosky.theta)}')
    print(f'{"ISO G_b: ":<24s}',  f'{value(m.fs.isosky.G_b):<10,.1f}', f'{pyunits.get_units(m.fs.isosky.G_b)}')
    print(f'{"ISO G: ":<24s}',  f'{value(m.fs.isosky.G):<10,.1f}', f'{pyunits.get_units(m.fs.isosky.G)}')
    print(f'{"ISO G_T: ":<24s}',  f'{value(m.fs.isosky.G_T):<10,.1f}', f'{pyunits.get_units(m.fs.isosky.G_T)}')
    print(f'{"ISO G_trans: ":<24s}',  f'{value(m.fs.isosky.G_trans):<10,.1f}', f'{pyunits.get_units(m.fs.isosky.G_trans)}')
    
def initialize_fpc(m):
    # Initial values for Flat Plate Collector
    m.fs.FPC.area_coll.set_value(1)
    m.fs.FPC.FRta.set_value(0.689)
    m.fs.FPC.FRUL.set_value(3.85)
    m.fs.FPC.iam.set_value(0.2)
    m.fs.FPC.mdot_test.set_value(0.045528)
    m.fs.FPC.cp_test.set_value(3400)  # specific heat of glycol [J/kg-K]
    m.fs.FPC.cp_use.set_value(3400)  # specific heat of glycol [J/kg-K]
    m.fs.FPC.ncoll.set_value(1)
    m.fs.FPC.pump_watts.set_value(45)
    m.fs.FPC.pump_eff.set_value(0.85)
    m.fs.FPC.T_amb.set_value(12)  # default SAM model at noon on Jan. 1
    m.fs.FPC.T_in.set_value(38.2)  # default SAM model at noon on Jan. 1
    m.fs.FPC.G_trans.set_value(m.fs.isosky.G_trans)  # default SAM model at noon on Jan. 1

def initialize_vagmd(m):
    # Initial values for Membrane Distillation
    m.fs.VAGMD.STEC = 100
    m.fs.VAGMD.permeate_flow_rate.fix(5e-5)

def initialize_tes(m):
    # initial values for Thermal Storage
    m.fs.TES.storage_eff.set_value(0.95)
    m.fs.TES.supply_eff.set_value(0.95)
    m.fs.TES.dt.set_value(1) # hr


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
    print("\nDegrees of Freedom: ",degrees_of_freedom(m),'\n')
    m.fs.FPC.initialize(outlvl=outlvl)
    m.fs.TES.initialize(outlvl=outlvl)
    # m.fs.VAGMD.initialize(outlvl=outlvl)

    return 

if __name__ == "__main__":
    m = build_flat_plate_flowsheet()
    fix_dof_and_initialize(m)
    results = solver.solve(m)

    print(f'{"ISO G_trans: ":<24s}',  f'{value(m.fs.isosky.G_trans):<10,.1f}', f'{pyunits.get_units(m.fs.isosky.G_trans)}')
    print(f'{"FPC G_trans: ":<24s}',  f'{value(m.fs.FPC.G_trans):<10,.1f}', f'{pyunits.get_units(m.fs.FPC.G_trans)}')
    print('\n')

    print(f'{"FPC Heat Gen: ":<24s}',  f'{value(m.fs.FPC.Q_useful):<10,.1f}', f'{pyunits.get_units(m.fs.FPC.Q_useful)}')
    print(f'{"System Heat Gen: ":<24s}',  f'{value(m.fs.heat_generation):<10,.1f}', f'{pyunits.get_units(m.fs.heat_generation)}')
    print(f'{"initial state of heat: ":<24s}',  f'{value(m.fs.TES.initial_state_of_charge):<10,.1f}', f'{pyunits.get_units(m.fs.TES.initial_state_of_charge)}')
    print(f'{"state of heat: ":<24s}',          f'{value(m.fs.TES.state_of_charge[0]):<10,.1f}', f'{pyunits.get_units(m.fs.TES.state_of_charge[0])}')
    print(f'{"Heat throughput: ":<24s}',        f'{value(m.fs.TES.heat_throughput[0]):<10,.1f}', f'{pyunits.get_units(m.fs.TES.heat_throughput[0])}')
    print(f'{"TES power: ":<24s}',            f'{value(m.fs.TES.heat_output):<10,.1f}', f'{pyunits.get_units(m.fs.TES.heat_output)}')
    print(f'{"TES heat: ":<24s}',           f'{value(m.fs.TES.heat_capacity):<10,.1f}', f'{pyunits.get_units(m.fs.TES.heat_capacity)}')
    # print(f'{"Heat generation: ":<24s}',   f'{value(m.fs.heat_generation):<10,.1f}', f'{pyunits.get_units(m.fs.heat_generation)}')