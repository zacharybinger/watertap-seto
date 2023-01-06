import pyomo.environ as pyo
from watertap.costing.util import register_costing_parameter_block
from watertap_contrib.seto.costing.util import (
    make_capital_cost_var,
    make_fixed_operating_cost_var,
    make_variable_operating_cost_var,
)


# def build_iph_trough_cost_param_block(blk):

#     costing = blk.parent_block()

#     blk.nothing = pyo.Var()


#     blk.fix_all_vars()


# @register_costing_parameter_block(
#     build_rule=build_iph_trough_cost_param_block, parameter_block_name="iph_trough"
# )
def cost_iph(blk):
    make_capital_cost_var(blk)
    make_fixed_operating_cost_var(blk)
    make_variable_operating_cost_var(blk)
    # Register flows
    # blk.config.flowsheet_costing_block.cost_flow(
    #     blk.unit_model.electricity, "electricity"
    # )
    blk.cap_cost = pyo.Var(initialize=0)
    blk.fixed_op = pyo.Var(initialize=0)
    blk.var_op = pyo.Var(initialize=0)

    blk.capital_cost_constraint = pyo.Constraint(expr=blk.capital_cost == blk.cap_cost)
    blk.fixed_operating_cost_constraint = pyo.Constraint(
        expr=blk.fixed_operating_cost == blk.fixed_op
    )
    blk.variable_operating_cost_constraint = pyo.Constraint(
        expr=blk.variable_operating_cost == blk.var_op
    )

    blk.costing_package.cost_flow(blk.unit_model.electricity, "electricity")

    blk.costing_package.cost_flow(blk.unit_model.heat, "heat")
