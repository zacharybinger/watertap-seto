import os
import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Param,
    TransformationFactory,
    assert_optimal_termination,
    units as pyunits,
    log10,
    Block,
    value,
    Objective,
    Constraint,
)
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.core.solvers.get_solver import get_solver
from idaes.models.unit_models import Product, Feed
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.scaling import (
    set_scaling_factor,
    calculate_scaling_factors,
    constraint_scaling_transform,
)
from idaes.core import UnitModelCostingBlock
from idaes.core.util.initialization import propagate_state

from watertap.property_models.NaCl_prop_pack import NaClParameterBlock
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice

from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.examples.flowsheets.RO_with_energy_recovery.RO_with_energy_recovery import (
    calculate_operating_pressure,
)
from watertap_contrib.reflo.analysis.net_metering.util import (
    display_ro_pv_results,
    display_pv_results,
)
from watertap_contrib.reflo.costing import (
    TreatmentCosting,
    EnergyCosting,
    REFLOCosting,
)
from watertap_contrib.reflo.solar_models.zero_order import Photovoltaic
from watertap_contrib.reflo.core import PySAMWaterTAP
from watertap_contrib.reflo.analysis.net_metering.RO import *

solver = get_solver()

absolute_path = os.path.dirname(__file__)
print(absolute_path)

def build_system():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = NaClParameterBlock()

    treatment = m.fs.treatment = Block()
    energy = m.fs.energy = Block()

    energy.pv = Photovoltaic()
    treatment.primary_pump = Pump(property_package=m.fs.properties)
    treatment.feed = Feed(property_package=m.fs.properties)
    treatment.product = Product(property_package=m.fs.properties)
    treatment.disposal = Product(property_package=m.fs.properties)
    
    treatment.ro = FlowsheetBlock(dynamic=False)
    
    build_ro(m,treatment.ro, number_of_stages=1)
    
    return m

def add_connections(m):

    m.fs.treatment.feed_to_primary_pump = Arc(
        source=m.fs.treatment.feed.outlet,
        destination=m.fs.treatment.primary_pump.inlet,
    )

    m.fs.treatment.primary_pump_to_ro = Arc(
        source=m.fs.treatment.primary_pump.outlet,
        destination=m.fs.treatment.ro.feed.inlet,
    )

    m.fs.treatment.ro_to_product = Arc(
        source=m.fs.treatment.ro.product.outlet,
        destination=m.fs.treatment.product.inlet,
    )
    m.fs.treatment.ro_to_disposal = Arc(
        source=m.fs.treatment.ro.disposal.outlet,
        destination=m.fs.treatment.disposal.inlet,
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

def add_constraints(m):
    m.fs.treatment.water_recovery = Var(
        initialize=0.5,
        bounds=(0, 0.99),
        domain=NonNegativeReals,
        units=pyunits.dimensionless,
        doc="System Water Recovery",
    )

    m.fs.treatment.feed_salinity = Var(
        initialize=35,
        bounds=(0, 2000),
        domain=NonNegativeReals,
        units=pyunits.dimensionless,
        doc="System Water Recovery",
    )

    m.fs.treatment.feed_flow_mass = Var(
        initialize=1,
        bounds=(0.000001, 1e6),
        domain=NonNegativeReals,
        units=pyunits.kg / pyunits.s,
        doc="System Feed Flowrate",
    )
    
    m.fs.treatment.perm_flow_mass = Var(
        initialize=1,
        bounds=(0.000001, 1e6),
        domain=NonNegativeReals,
        units=pyunits.kg / pyunits.s,
        doc="System Produce Flowrate",
    )

    m.fs.treatment.nacl_mass_constraint = Constraint(
        expr=m.fs.treatment.feed.flow_mass_phase_comp[0, "Liq", "NaCl"] * 1000
        == m.fs.treatment.feed_flow_mass * m.fs.treatment.feed_salinity
    )

    m.fs.treatment.h2o_mass_constraint = Constraint(
        expr=m.fs.treatment.feed.flow_mass_phase_comp[0, "Liq", "H2O"]
        == m.fs.treatment.feed_flow_mass * (1 - m.fs.treatment.feed_salinity / 1000)
    )

    m.fs.treatment.eq_water_recovery = Constraint(
        expr=m.fs.treatment.feed.properties[0].flow_vol * m.fs.treatment.water_recovery
        == m.fs.treatment.product.properties[0].flow_vol
    )

    m.fs.treatment.product.properties[0].mass_frac_phase_comp
    m.fs.treatment.feed.properties[0].conc_mass_phase_comp
    m.fs.treatment.product.properties[0].conc_mass_phase_comp
    m.fs.treatment.disposal.properties[0].conc_mass_phase_comp
    m.fs.treatment.primary_pump.control_volume.properties_in[0].conc_mass_phase_comp
    m.fs.treatment.primary_pump.control_volume.properties_out[0].conc_mass_phase_comp

    m.fs.treatment.ro.product.properties[0.0].conc_mass_phase_comp["Liq", "NaCl"].setlb(0)
    m.fs.treatment.product.properties[0.0].conc_mass_phase_comp["Liq", "NaCl"].setlb(0)

def add_costing(m):
    treatment = m.fs.treatment
    energy = m.fs.energy
    treatment.costing = TreatmentCosting()
    energy.costing = EnergyCosting()

    # energy.pv.costing = UnitModelCostingBlock(flowsheet_costing_block=energy.costing)
    treatment.ro.stage[1].module.costing = UnitModelCostingBlock(
        flowsheet_costing_block=treatment.costing
    )
    # treatment.erd.costing = UnitModelCostingBlock(
    #     flowsheet_costing_block=treatment.costing
    # )
    treatment.primary_pump.costing = UnitModelCostingBlock(
        flowsheet_costing_block=treatment.costing
    )

    treatment.costing.cost_process()
    # energy.costing.cost_process()

    # m.fs.sys_costing = REFLOCosting()
    # m.fs.sys_costing.add_LCOW(treatment.product.properties[0].flow_vol)
    # m.fs.sys_costing.add_specific_electric_energy_consumption(
    #     treatment.product.properties[0].flow_vol
    # )

    # treatment.costing.initialize()
    # energy.costing.initialize()



def release_constraints(m):
    # These are turned off or relaxed to allow for ultrapure water, but this does not
    # mean that these assumptions are valid. These are relaxed for the purpose of allowing
    # the model to solve, but should be highlighted to show why infeasible solutions arise.

    # Release constraints for system
    m.fs.treatment.ro.product.properties[0.0].conc_mass_phase_comp["Liq", "NaCl"].setlb(0)
    m.fs.treatment.product.properties[0.0].conc_mass_phase_comp["Liq", "NaCl"].setlb(0)

def set_inlet_conditions(m, Qin=None, Cin=None, water_recovery=None, primary_pump_pressure=80e5):
    """Sets operating condition for the PV-RO system

    Args:
        m (obj): Pyomo model
        flow_in (float, optional): feed volumetric flow rate [m3/s]. Defaults to 1e-2.
        conc_in (int, optional): solute concentration [g/L]. Defaults to 30.
        water_recovery (float, optional): water recovery. Defaults to 0.5.
    """
    print(f'\n{"=======> SETTING OPERATING CONDITIONS <=======":^60}\n')
    solver = get_solver()    
    if Qin is None:
        m.fs.treatment.feed_flow_mass.fix(1)
    else:
        m.fs.treatment.feed_flow_mass.fix(Qin)

    if Cin is None:
        m.fs.treatment.feed_salinity.fix(10)
    else:
        m.fs.treatment.feed_salinity.fix(Cin)

    if water_recovery is not None:
        m.fs.treatment.water_recovery.fix(water_recovery)
        m.fs.treatment.primary_pump.control_volume.properties_out[0].pressure.unfix()
    else:
        m.fs.treatment.water_recovery.unfix()
        m.fs.treatment.primary_pump.control_volume.properties_out[0].pressure.fix(primary_pump_pressure)

    # iscale.set_scaling_factor(m.fs.treatment.perm_flow_mass, 1)
    iscale.set_scaling_factor(m.fs.treatment.feed_flow_mass, 1)
    iscale.set_scaling_factor(m.fs.treatment.feed_salinity, 1)

    feed_temperature = 273.15 + 20
    pressure_atm = 101325
    supply_pressure = 2.7e5

    # initialize feed
    m.fs.treatment.feed.pressure[0].fix(supply_pressure)
    m.fs.treatment.feed.temperature[0].fix(feed_temperature)

    m.fs.treatment.primary_pump.efficiency_pump.fix(0.85)
    iscale.set_scaling_factor(m.fs.treatment.primary_pump.control_volume.work, 1e-3)

    m.fs.treatment.feed.properties[0].flow_vol_phase["Liq"]
    m.fs.treatment.feed.properties[0].mass_frac_phase_comp["Liq", "NaCl"]

    m.fs.treatment.feed.flow_mass_phase_comp[0, "Liq", "NaCl"].value = (
        m.fs.treatment.feed_flow_mass.value * m.fs.treatment.feed_salinity.value / 1000
    )
    m.fs.treatment.feed.flow_mass_phase_comp[
        0, "Liq", "H2O"
    ].value = m.fs.treatment.feed_flow_mass.value * (1 - m.fs.treatment.feed_salinity.value / 1000)

    scale_flow = calc_scale(m.fs.treatment.feed.flow_mass_phase_comp[0, "Liq", "H2O"].value)
    scale_tds = calc_scale(m.fs.treatment.feed.flow_mass_phase_comp[0, "Liq", "NaCl"].value)

    # operating_pressure = calculate_operating_pressure(
    # feed_state_block=m.fs.treatment.feed.properties[0],
    # over_pressure=0.15,
    # water_recovery=0.8,
    # NaCl_passage=0.01,
    # solver=solver,
    # )

    # operating_pressure_psi = pyunits.convert(
    #     operating_pressure * pyunits.Pa, to_units=pyunits.psi
    # )()
    # operating_pressure_bar = pyunits.convert(
    #     operating_pressure * pyunits.Pa, to_units=pyunits.bar
    # )()
    # print(
    #     f"\nOperating Pressure Estimate = {round(operating_pressure_bar, 2)} bar = {round(operating_pressure_psi, 2)} psi\n"
    # )

    # REVIEW: Make sure this is applied in the right place
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 10**-scale_flow, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 10**-scale_tds, index=("Liq", "NaCl")
    )

    assert_units_consistent(m)

def set_operating_conditions(m):
    # Set inlet conditions and operating conditions for each unit
    set_inlet_conditions(m, Qin=0.031976*1000, Cin=30, primary_pump_pressure=50e5)
    # set_inlet_conditions(m, Qin=0.031976*1000, Cin=0.004, primary_pump_pressure=10e5)
    set_ro_system_operating_conditions(m, m.fs.treatment.ro, mem_area=[37*9*6, 37*5*6])

def init_system(m, verbose=True, solver=None):
    if solver is None:
        solver = get_solver()

    optarg = solver.options

    print("\n\n-------------------- INITIALIZING SYSTEM --------------------\n\n")
    print(f"System Degrees of Freedom: {degrees_of_freedom(m)}")

    m.fs.treatment.feed.initialize(optarg=optarg)
    propagate_state(m.fs.treatment.feed_to_primary_pump)

    m.fs.treatment.primary_pump.initialize(optarg=optarg)
    propagate_state(m.fs.treatment.primary_pump_to_ro)

    init_ro_system(m, m.fs.treatment.ro)

    propagate_state(m.fs.treatment.ro_to_product)
    propagate_state(m.fs.treatment.ro_to_disposal)

    m.fs.treatment.product.initialize(optarg=optarg)
    m.fs.treatment.disposal.initialize(optarg=optarg)

def solve(model, solver=None, tee=True, raise_on_failure=True):
    # ---solving---
    if solver is None:
        solver = get_solver()

    print("\n--------- SOLVING ---------\n")

    results = solver.solve(model, tee=tee)

    if check_optimal_termination(results):
        print("\n--------- OPTIMAL SOLVE!!! ---------\n")
        return results
    msg = (
        "The current configuration is infeasible. Please adjust the decision variables."
    )
    if raise_on_failure:
        # debug(model, solver=solver, automate_rescale=False, resolve=False)
        # debug(model, solver=solver, automate_rescale=False, resolve=False)
        # check_jac(model)
        print_close_to_bounds(model)
        raise RuntimeError(msg)
    else:
    #     print(msg)
    #     # debug(model, solver=solver, automate_rescale=False, resolve=False)
    #     # check_jac(model)
        return results


if __name__ == "__main__":
    m = build_system()
    display_ro_system_build(m, decend = False)

    add_connections(m)
    add_constraints(m)
    add_costing(m)
    # set_operating_conditions(m)

    # # Initialize system, ititialization routines for each unit in definition for init_system
    # init_system(m)

    # display_flow_table(m, m.fs.treatment.ro)

    # # Solve system and display results
    # solve(m)
    # display_flow_table(m, m.fs.treatment.ro)
    # display_system_metrics(m, m.fs.treatment.ro)