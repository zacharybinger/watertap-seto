# General python imports
import numpy as np
import pandas as pd
import logging
from collections import deque
from os.path import join, dirname
# Pyomo imports
from pyomo.environ import Set, Expression, value, Objective
import datetime
# IDAES imports
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
# Flowsheet function imports
from watertap_contrib.seto.analysis.multiperiod.flat_plate_flowsheet import (
    build_flat_plate_flowsheet,
    fix_dof_and_initialize,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
import seaborn as sns
from idaes.core.surrogate.pysmo_surrogate import PysmoRBFTrainer, PysmoSurrogate
from pyomo.environ import Var, value, units as pyunits

__author__ = "Zachary Binger"

_log = idaeslog.getLogger(__name__)
solver = get_solver()

# def eval_FPC(m, iam):
#     m.fs.FPC.iam.set_value(iam)

#     return 

def get_fpc_md_tes_variable_pairs(t1, t2):
    """
    This function returns pairs of variables that need to be connected across two time periods

    Args:
        t1: current time block
        t2: next time block

    Returns:
        None
    """
    return [
        (t1.fs.TES.state_of_charge[0], t2.fs.TES.initial_state_of_charge),
        (t1.fs.TES.heat_throughput[0], t2.fs.TES.initial_heat_throughput),
        (t1.fs.TES.heat_output, t2.fs.TES.heat_output),
        (t1.fs.TES.heat_capacity, t2.fs.TES.heat_capacity),
        ]

def unfix_dof(m):
    """
    This function unfixes a few degrees of freedom for optimization

    Args:
        m: object containing the integrated nuclear plant flowsheet

    Returns:
        None
    """
    # m.fs.TES.nameplate_energy.unfix()
    # m.fs.TES.nameplate_energy.fix(3000)
    m.fs.TES.heat_capacity.unfix()
    m.fs.TES.heat_output.unfix()

    return

heat_generation = [0,0,0,0,0,0,0,0,10,12,14,18,24,32,24,18,14,12,10,0,0,0,0,0]

def create_multiperiod_fpc_md_tes_model(
        n_time_points= 24,
        md_capacity = 5, # m3/day
        md_heat_req = 1800, # kW
        cost_tes_power = 75, # $/kW
        cost_tes_energy = 50, # $/kWh      
        heat_price = 0.15,
    ):
    pass

    """
    This function creates a multi-period md tes flowsheet object. This object contains 
    a pyomo model with a block for each time instance.

    Args:
        n_time_points: Number of time blocks to create

    Returns:
        Object containing multi-period vagmd batch flowsheet model
    """
    mp = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=build_flat_plate_flowsheet,
        linking_variable_func=get_fpc_md_tes_variable_pairs,
        initialization_func=fix_dof_and_initialize,
        unfix_dof_func=unfix_dof,
        outlvl=logging.WARNING,
    )

    flowsheet_options={ t: { 
                            "heat_price": heat_price,
                            "md_capacity": md_capacity, 
                            "md_heat_req": md_heat_req,
                            "day":51,
                            "hour":t} 
                            for t in range(n_time_points)
    }

    # create the multiperiod object
    mp.build_multi_period_model(
        model_data_kwargs=flowsheet_options,
        flowsheet_options={ "md_capacity": md_capacity, 
                            "md_heat_req": md_heat_req},
        # initialization_options=None,
        # unfix_dof_options=None,
        )
    
    mp.blocks[0].process.fs.TES.initial_state_of_charge.fix(0)

    @mp.Expression(doc="Heat storage cost")
    def tes_cost(b):
        return ( 0.168 * # capital recovery factor
            (cost_tes_power * b.blocks[0].process.fs.TES.heat_output
            +cost_tes_energy * b.blocks[0].process.fs.TES.heat_capacity))
        
    # Add fpc cost function
    @mp.Expression(doc="MD cost")
    def md_cost(b):
        return (
            1040 * b.blocks[0].process.fs.VAGMD.membrane_area * 0.168 # Annualized CAPEX
            + 9 * b.blocks[0].process.fs.VAGMD.membrane_area)          # OPEX

    # Total cost
    @mp.Expression(doc='total cost')
    def total_cost(b):
        # The annualized capital cost is evenly distributed to the multiperiod
        return (
            (b.tes_cost + b.md_cost) / 365 / 24 * n_time_points
            + sum([b.blocks[i].process.grid_cost for i in range(n_time_points)])
        )
    
    # LCOW
    @mp.Expression(doc='total cost')
    def LCOW(b):
        # LCOW from MD: 0.45
        return (
            b.total_cost / md_capacity / 24 * n_time_points + 0.45
        )   

    # Set objective
    mp.obj = Objective(expr=mp.LCOW)

    return mp

def create_plot(mp, idx, norm=False):
    # Create diagrams
    # plt.clf()
    colors=['#235789', '#4A7C59', '#F1A208']
    n = 24
    titles = ['Summer','Winter','Spring', 'Fall']
    hour = [i for i in range(1,n+1)]
    iso_G_trans = np.array([value(mp.blocks[i].process.fs.isosky.G_trans) for i in range(n)])
    fpc_Q_useful = np.array([value(mp.blocks[i].process.fs.FPC.Q_useful) for i in range(n)])
    tes_state = np.array([value(mp.blocks[i].process.fs.TES.state_of_charge[0]) for i in range(n)])
    heat_gen = np.array([value(mp.blocks[i].process.fs.heat_generation) for i in range(n)])
    hour_in = np.array([value(mp.blocks[i].process.fs.hour) for i in range(n)])
    heat_curtail = np.array([value(mp.blocks[i].process.fs.curtailment) for i in range(n)])
    heat_price = np.array([value(mp.blocks[i].process.fs.heat_price) for i in range(n)])
    md_demand = np.array([value(mp.blocks[i].process.fs.VAGMD.thermal_power) for i in range(n)])
    axes[idx].plot(hour, iso_G_trans, 'tab:orange', label='Irradiance (W/m**2)')
    axes[idx].plot(hour, fpc_Q_useful, 'tab:blue', label='Heat (W)')
    axes[idx].plot(hour, heat_gen, 'k', label = 'Heat Generation (kWh)')
    axes[idx].plot(hour, hour_in, 'tab:green', label = 'Day')
    # axes[idx].plot(hour, heat_curtail, 'g', label = 'Heat Curtailment (kWh)')
    # axes[idx].plot(hour, md_demand, '--', color='tab:red', label = 'VAGM Heat Demand (kWh)')
    # axes[idx].plot(hour, heat_gen + heat_curtail , 'k', label = 'Total Energy Flow (kWh)')
    # axes[idx].vlines(x=[day*24 for day in range(7)],ymin=0,ymax=6000,linestyle='--',color='black')
    # axes[idx].set_ylim([0,np.array(tes_state).max()])
    axes[idx].set_xlim([1,n])
    axes[idx].set_ylabel('  Energy (kWh)', loc='center', fontsize=16)
    axes[idx].set_xlabel('Operating Hours', fontsize=16)
    
    axes[idx].set_title(titles[idx], loc='center', x=-0.09, y=0.5, rotation=90, fontweight='bold', ha='center', va='center', fontsize=16)
    axes[idx].tick_params(axis="x", labelsize=16)
    axes[idx].tick_params(axis="y", labelsize=16)
    ax3 = axes[idx].twinx()
    ax3.plot(hour, heat_price,'--',label='Grid Price ($/kWh)')
    ax3.set_ylabel('Grid Price ($/kWh)', ha='center', va='center', fontsize=16, labelpad=20)
    # ax3.set_ylim([0,0.3])
    
    ax3.tick_params(axis="y", labelsize=16)

    # fpc_to_md = np.array([value(mp.blocks[i].process.fs.fpc_to_md) for i in range(n)])
    # fpc_to_tes = np.array([value(mp.blocks[i].process.fs.TES.heat_in[0]) for i in range(n)])
    # tes_to_md = np.array([value(mp.blocks[i].process.fs.TES.heat_out[0]) for i in range(n)])
    # grid_to_md = np.array([value(mp.blocks[i].process.fs.grid_to_md) for i in range(n)])
    # labels=["FPC to MD", "TES to MD", "Grid to MD", "FPC to TES"]
    # norm_labels=["FPC to MD", "TES to MD", "FPC to TES", "Grid to MD"]
    # axes[idx].plot(hour, fpc_to_tes, 'b', label = 'FPC to TES (kWh)')

    # leg1 = axes[idx].legend(loc="lower left", frameon = False, bbox_to_anchor=(0, 1.0, 0.65, 1),
    #     ncols=5, mode="expand", borderaxespad=0.)
    # leg2 = ax3.legend(loc="lower left", frameon = False, bbox_to_anchor=(0.66, 1.0, 0.15, 1),
    #     ncols=1, mode="expand", borderaxespad=0.)
    
    # frames = []
    # for label in labels:
    #     frames.append(pd.DataFrame([hour, fpc_to_md, [label]*len(fpc_to_md)]).transpose())
    # df2 = pd.concat(frames)
    # df2.columns = ['Hour', 'State', 'Type']

    # df = pd.DataFrame([hour, fpc_to_md, fpc_to_tes, tes_to_md, grid_to_md]).transpose()
    # df.columns = ["Hour", "FPC to MD", "FPC to TES", "TES to MD", "Grid to MD"]
    # features = ["FPC to MD", "FPC to TES", "TES to MD", "Grid to MD"]
    # df['Total'] = df["FPC to MD"] + df["TES to MD"] + df["Grid to MD"] + df["FPC to TES"]
    # # # df.to_csv('/Users/zbinger/watertap-seto/src/watertap_contrib/seto/analysis/multiperiod/data_files/sim_results.csv')
    # if norm == True:
    #     for feature in features:
    #         df[feature] = (df[feature]/df['Total'])
    #     axes2[idx].stackplot(hour, 100*df["FPC to MD"],  100*df["TES to MD"], 100*df["FPC to TES"], 100*df["Grid to MD"], baseline='zero', labels=norm_labels, alpha=1, ec='white')
    #     axes2[idx].set_ylabel('Heat Load %', loc='center', fontsize=16)
    # else:
    #     axes2[idx].stackplot(hour, df["FPC to MD"],  df["TES to MD"], df["Grid to MD"], baseline='zero', colors=['#1f77b4','#ff7f0e','#d62728'], labels=labels, alpha=1, ec='white')
    #     axes2[idx].plot(hour, df["FPC to TES"], label="FPC to TES", color='#f78d02')
    #     axes2[idx].set_ylabel('  Power (kW)', loc='center', fontsize=16)
    # # df.to_csv('/Users/zbinger/watertap-seto/src/watertap_contrib/seto/analysis/multiperiod/data_files/sim_results_norm.csv')
    # # ax2.set_xlabel('Hour (June 18th)')

    # axes2[idx].set_xlabel('Operation Hours', fontsize=16)
    # leg3 = axes2[idx].legend(loc="lower left", frameon = False, bbox_to_anchor=(0, 1.0, 0.35, 1),
    #     ncols=4, mode="expand", borderaxespad=0.)
    # if norm == True:
    #     axes2[idx].set_ylim([0,100])
    #     axes2[idx].yaxis.set_major_formatter(mtick.PercentFormatter()) 
    #     axes2[idx].vlines(x=[day*24 for day in range(7)],ymin=0,ymax=100,linestyle='--',color='black')
    # axes2[idx].set_xlim([1,n])
    # axes2[idx].set_title(titles[idx], loc='center', x=-0.08, y=0.5, rotation=90, fontweight='bold', ha='center', va='center', fontsize=16)
    # axes2[idx].tick_params(axis="x", labelsize=16)
    # axes2[idx].tick_params(axis="y", labelsize=16)
    # # ax4 = axes2[idx].twinx()
    # # ax4.plot(hour, heat_prices,linestyle='dotted', color='k',label='Grid Price')
    # # ax4.set_ylabel('Grid Price ($/kWh)', ha='center', va='center', fontsize=16, labelpad=20)
    # # ax4.set_ylim([0,0.3])
    # # leg4 = ax4.legend(loc="lower left", frameon = False, bbox_to_anchor=(0.37, 1.0, 0.15, 1),
    # #     ncols=1, mode="expand", borderaxespad=0.)
    # # ax4.tick_params(axis="y", labelsize=16)

if __name__ == "__main__":
    fig,  axes= plt.subplots(2, figsize=(20,10))
    # fig2,  axes2= plt.subplots(4, figsize=(20,10))
    mp = create_multiperiod_fpc_md_tes_model()
    results = solver.solve(mp)
    create_plot(mp, 0, norm=True)
    fig.tight_layout()
    plt.show()