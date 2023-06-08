import pyomo.environ as pyo
from watertap.costing.util import register_costing_parameter_block
from watertap_contrib.seto.costing.util import (
    make_capital_cost_var,
    make_fixed_operating_cost_var,
    make_variable_operating_cost_var,
)

def build_battery_cost_param_block(blk):

    costing = blk.parent_block() 

    blk.cost_per_kw = pyo.Param(
        initialize = 269.41,
        units = costing.base_currency/ pyo.units.kW,
        bounds = (0,None),
        mutable = True,
        doc = 'Battery cost per kW DC'
    )

    blk.cost_per_kwh= pyo.Param(
        initialize = 282,
        units = costing.base_currency/ (pyo.units.kWh),
        bounds = (0,None),
        mutable = True,
        doc = 'Battery cost per kWh DC'
    )

    blk.contingency_frac_direct_cost = pyo.Param(
        initialize = 282,
        units = costing.base_currency/ (pyo.units.kW * pyo.units.h),
        bounds = (0,None),
        mutable = True,
        doc = 'Fraction of direct costs for contigency'
    )

    blk.sales_tax_frac = pyo.Param(
        initialize = 0.05,
        units = pyo.units.dimensionless,
        bounds = (0,None),
        mutable = True,
        doc = 'Fraction of direct costs for sales tax'
    )

    blk.cost_frac_indirect = pyo.Param(
        initialize = 0.135,
        units = pyo.units.dimensionless,
        bounds = (0,None),
        mutable = True,
        doc = 'Fraction of direct costs for indirect costs for engineering, permitting and other EPC costs'
    )

    blk.fixed_operating_by_capacity = pyo.Var(
        initialize = 15,
        units=costing.base_currency / (pyo.units.kWh),
        bounds=(0, None),
        doc= 'Fixed operating cost of battery system per kWh generated'
    )


# Battery replacement cost can be included?

@register_costing_parameter_block(
    build_rule=build_battery_cost_param_block, parameter_block_name="battery"
)

def cost_battery(blk):
    
    battery_params = blk.costing_package.battery
    make_capital_cost_var(blk)
    make_variable_operating_cost_var(blk)
    make_fixed_operating_cost_var(blk)

    blk.direct_cost = pyo.Var(
        initialize=0,
        units=blk.config.flowsheet_costing_block.base_currency,
        bounds=(0, None),
        doc="Direct costs of PV system",
    )

    blk.indirect_cost = pyo.Var(
        initialize=0,
        units=blk.config.flowsheet_costing_block.base_currency,
        bounds=(0, None),
        doc="Indirect costs of PV system",
    )

    blk.sales_tax = pyo.Var(
        initialize=0,
        units=blk.config.flowsheet_costing_block.base_currency,
        bounds=(0, None),
        doc="Sales tax for PV system",
    )

    blk.battery_capacity = pyo.Var(
        initialize=0,
        units=pyo.units.kWh,
        bounds=(0, None),
        doc="Battery capacity",
    )

    blk.battery_power = pyo.Var(
        initialize=0,
        units=pyo.units.kW,
        bounds=(0, None),
        doc="Battery power",
    )

    blk.direct_cost_constraint = pyo.Constraint(
        expr=blk.direct_cost 
        == (blk.battery_capacity*battery_params.cost_per_kwh + 
            blk.battery_power*battery_params.cost_per_kw) * (1 + battery_params.contingency_frac_direct_cost)
    )
    
    blk.indirect_cost_constraint = pyo.Constraint(
        expr=blk.indirect_cost == blk.direct_cost * battery_params.cost_frac_indirect

    )
        
    blk.sales_tax_constraint = pyo.Constraint(
        expr = blk.sales_tax 
        == blk.direct_cost * battery_params.sales_tax_frac
    )

    blk.capital_cost_constraint = pyo.Constraint(
        expr=blk.capital_cost == blk.direct_cost + blk.indirect_cost + blk.sales_tax
    )

    blk.fixed_operating_cost_constraint = pyo.Constraint(
        expr=blk.fixed_operating_cost
        == battery_params.fixed_operating_by_capacity
        * blk.battery_capacity
    )