
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
from watertap_contrib.seto.analysis.multiperiod.PV_RO_battery_flowsheet import (
    build_pv_battery_flowsheet,
    fix_dof_and_initialize,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
import seaborn as sns
from idaes.core.surrogate.pysmo_surrogate import PysmoRBFTrainer, PysmoSurrogate

__author__ = "Zhuoran Zhang, Mukta Hardikar, Zachary Binger"

_log = idaeslog.getLogger(__name__)
solver = get_solver()

def load_surrogate(surrogate_filename=None):
    return PysmoSurrogate.load_from_file(surrogate_filename)

def eval_surrogate(surrogate, design_size, Hour, Day):
    input = pd.DataFrame.from_dict([{'design_size':design_size, 'Hour':Hour, 'Day':Day}], orient='columns')
    hourly_gen = surrogate.evaluate_surrogate(input)
    return hourly_gen.values[0][0]

def get_pv_ro_variable_pairs(t1, t2):
    """
    This function returns pairs of variables that need to be connected across two time periods

    Args:
        t1: current time block
        t2: next time block

    Returns:
        None
    """
    return [
        (t1.fs.battery.state_of_charge[0], t2.fs.battery.initial_state_of_charge),
        (t1.fs.battery.energy_throughput[0], t2.fs.battery.initial_energy_throughput),
        (t1.fs.battery.nameplate_power, t2.fs.battery.nameplate_power),
        (t1.fs.battery.nameplate_energy, t2.fs.battery.nameplate_energy),
        (t1.fs.pv.size, t2.fs.pv.size)]

def unfix_dof(m):
    """
    This function unfixes a few degrees of freedom for optimization

    Args:
        m: object containing the integrated nuclear plant flowsheet

    Returns:
        None
    """
    m.fs.battery.nameplate_energy.unfix()
    m.fs.battery.nameplate_power.unfix()
    # m.fs.battery.initial_state_of_charge.unfix()
    # m.fs.battery.initial_energy_throughput.unfix()
    return

# PV surrogate output for 4 select days
file_path = '/Users/zbinger/watertap-seto/src/watertap_contrib/seto/analysis/multiperiod'

# Arbitrary electricity costs
elec_price_df = pd.read_csv(file_path +'/data_files/elec_price.csv',index_col='time (h)')
elec_price = np.array(elec_price_df['elec_price'].values)
elec_price = np.append(elec_price,elec_price)
elec_price = np.append(elec_price,elec_price)

electric_tiers = {'Tier 1':0.19825,'Tier 2':0.06124,'Tier 3':0.24445,'Tier 4':0.06126}

def get_rep_weeks():
    year_start = datetime.datetime(year=2020, month=1, day=1, hour=0, minute=0)
    sum_solstice = datetime.datetime(year=2020, month=6, day=20, hour=0, minute=0)
    win_solstice = datetime.datetime(year=2020, month=12, day=21, hour=0, minute=0)
    ver_eq = datetime.datetime(year=2020, month=3, day=20, hour=0, minute=0)
    aut_eq = datetime.datetime(year=2020, month=9, day=22, hour=0, minute=0)
    key_days = [sum_solstice, win_solstice, ver_eq, aut_eq]
    return [(x-year_start).days*24 for x in key_days], key_days

def get_elec_tier(day, time):
    if day.weekday():
        print('Weekday')
    else:
        print('Weekend')




def create_multiperiod_pv_battery_model(
        n_time_points= 7*24,
        ro_capacity = 6000, # m3/day
        ro_elec_req = 944.3, # kW
        cost_battery_power = 75, # $/kW
        cost_battery_energy = 50, # $/kWh      
        elec_price = elec_price,
        surrogate = None,
        start_date = None
    ):
    
    """
    This function creates a multi-period pv battery flowsheet object. This object contains 
    a pyomo model with a block for each time instance.

    Args:
        n_time_points: Number of time blocks to create

    Returns:
        Object containing multi-period vagmd batch flowsheet model
    """
    mp = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=build_pv_battery_flowsheet,
        linking_variable_func=get_pv_ro_variable_pairs,
        initialization_func=fix_dof_and_initialize,
        unfix_dof_func=unfix_dof,
        outlvl=logging.WARNING,
    )
    if surrogate == None:
        surrogate = load_surrogate(surrogate_filename='/Users/zbinger/watertap-seto/src/watertap_contrib/seto/analysis/net_metering/pysam_data/pv_Spring_Eq_surrogate_week.json')
        
    flowsheet_options={ t: { 
                            "pv_gen": eval_surrogate(surrogate, ro_elec_req, t%24, t//24),
                            "elec_price": elec_price[t],
                            "ro_capacity": ro_capacity, 
                            "ro_elec_req": ro_elec_req} 
                            for t in range(n_time_points)
    }

    # create the multiperiod object
    mp.build_multi_period_model(
        model_data_kwargs=flowsheet_options,
        flowsheet_options={ "ro_capacity": ro_capacity, 
                            "ro_elec_req": ro_elec_req},
        # initialization_options=None,
        # unfix_dof_options=None,
        )

    # initialize the beginning status of the system
    mp.blocks[0].process.fs.battery.initial_state_of_charge.fix(0)
    # mp.blocks[23].process.fs.battery.initial_state_of_charge.fix(10)
    # mp.blocks[47].process.fs.battery.initial_state_of_charge.fix(10)
    # mp.blocks[71].process.fs.battery.initial_state_of_charge.fix(10)
    # mp.blocks[95].process.fs.battery.initial_state_of_charge.fix(10)
    # mp.blocks[0].process.fs.battery.initial_energy_throughput.fix(0)

    # Add battery cost function
    @mp.Expression(doc="battery cost")
    def battery_cost(b):
        return ( 0.096 * # capital recovery factor
            (cost_battery_power * b.blocks[0].process.fs.battery.nameplate_power
            +cost_battery_energy * b.blocks[0].process.fs.battery.nameplate_energy))
        
    # Add PV cost function
    @mp.Expression(doc="PV cost")
    def pv_cost(b):
        return (
            1040 * b.blocks[0].process.fs.pv.size * 0.096 # Annualized CAPEX
            + 9 * b.blocks[0].process.fs.pv.size)          # OPEX

    # Total cost
    @mp.Expression(doc='total cost')
    def total_cost(b):
        # The annualized capital cost is evenly distributed to the multiperiod
        return (
            (b.battery_cost + b.pv_cost) / 365 / 24 * n_time_points
            + sum([b.blocks[i].process.grid_cost for i in range(n_time_points)])
        )

    # LCOW
    @mp.Expression(doc='total cost')
    def LCOW(b):
        # LCOW from RO: 0.45
        return (
            b.total_cost / ro_capacity / 24 * n_time_points + 0.45
        )   

    # Set objective
    mp.obj = Objective(expr=mp.LCOW)

    return mp

def create_plot(mp, idx):
    # Create diagrams
    # plt.clf()
    colors=['#235789', '#4A7C59', '#F1A208']
    n = 7*24
    titles = ['Summer','Winter','Spring', 'Fall']
    hour = [i for i in range(1,97)]
    battery_state = [value(mp.blocks[i].process.fs.battery.state_of_charge[0]) for i in range(n)]
    pv_gen = [value(mp.blocks[i].process.fs.pv.elec_generation) for i in range(n)]
    pv_curtail = [value(mp.blocks[i].process.fs.curtailment) for i in range(n)]

    axes[idx].plot(hour, battery_state, 'r', label='Battery state (kWh)')
    axes[idx].plot(hour, pv_gen, 'k', label = 'PV generation (kWh)')
    axes[idx].plot(hour, pv_curtail, 'g', label = 'PV curtailment (kWh)')
    axes[idx].vlines(x=[24,48,72,96],ymin=0,ymax=6000,linestyle='--',color='black')
    axes[idx].set_ylim([0,6000])
    axes[idx].set_xlim([1,100])
    axes[idx].set_ylabel('  Energy (kWh)', loc='center')
    axes[idx].set_xlabel('Operation Hours')
    axes[idx].legend(loc="upper left", frameon = False)
    axes[idx].set_title(titles[idx], loc='center', x=-0.09, y=0.5, rotation=90, fontweight='bold', ha='center', va='center')

    ax3 = axes[idx].twinx()
    ax3.plot(hour, elec_price,'--',label='Grid Price')
    ax3.set_ylabel('Grid Price ($/kWh)', ha='center', va='center')
    ax3.set_ylim([0,3])
    ax3.legend(loc="upper right", frameon = False, fontsize = 'small')

    pv_to_ro = np.array([value(mp.blocks[i].process.fs.pv_to_ro) for i in range(n)])
    battery_to_ro = np.array([value(mp.blocks[i].process.fs.battery.elec_out[0]) for i in range(n)])
    grid_to_ro = np.array([value(mp.blocks[i].process.fs.grid_to_ro) for i in range(n)])
    labels=["PV to RO", "Battery to RO", "Grid to RO"]

    # frames = []
    # for label in labels:
    #     frames.append(pd.DataFrame([hour, pv_to_ro, [label]*len(pv_to_ro)]).transpose())
    # df2 = pd.concat(frames)
    # df2.columns = ['Hour', 'State', 'Type']

    # df = pd.DataFrame([hour, pv_to_ro, battery_to_ro, grid_to_ro]).transpose()
    # df.columns = ["Hour", "PV to RO", "Battery to RO", "Grid to RO"]

    # ax2.set_xlabel('Hour (June 18th)')
    axes2[idx].set_ylabel('  Load %', loc='center')
    axes2[idx].set_xlabel('Operation Hours')
    axes2[idx].stackplot(hour, 100*pv_to_ro/pv_to_ro.max(), 100*battery_to_ro/battery_to_ro.max(), 100*grid_to_ro/grid_to_ro.max(), baseline='zero', labels=labels, alpha=1, ec='white')
    # axes2[idx].stackplot(hour, 100*pv_to_ro/pv_to_ro.max(), 100*battery_to_ro/battery_to_ro.max(), 100*grid_to_ro/grid_to_ro.max(), baseline='zero', labels=labels, colors=colors, alpha=1, ec='white')
    axes2[idx].legend(loc="upper left", frameon = False)
    axes2[idx].yaxis.set_major_formatter(mtick.PercentFormatter()) 
    axes2[idx].vlines(x=[24,48,72,96],ymin=0,ymax=1000,linestyle='--',color='black')
    axes2[idx].set_ylim([0,100])
    axes2[idx].set_xlim([1,100])
    axes2[idx].set_title(titles[idx], loc='center', x=-0.08, y=0.5, rotation=90, fontweight='bold', ha='center', va='center')
    
if __name__ == "__main__":
    rep_days, key_days = get_rep_weeks()
    print(key_days[0])
    get_elec_tier(key_days[0], 4)
    fig,  axes= plt.subplots(4, figsize=(14,10))
    fig2,  axes2= plt.subplots(4, figsize=(14,10))
    for idx, period in enumerate(['Summer Solstice','Winter Solstice','Spring Eq', 'Fall Eq']):
        surr = load_surrogate(surrogate_filename=join('/Users/zbinger/watertap-seto/src/watertap_contrib/seto/analysis/net_metering/pysam_data/', "pv_"+period.replace(" ","_")+"_surrogate_week.json"))
        mp = create_multiperiod_pv_battery_model(surrogate = surr, start_date = key_days[idx])
    #     results = solver.solve(mp)
    #     create_plot(mp, idx)
    #     fig.tight_layout()
    #     fig2.tight_layout()
        
    # # for i in range(96):
    # #     print(f'battery status at hour: {i}', value(mp.blocks[i].process.fs.battery.state_of_charge[0]))    
    # #     print('pv gen(kW): ', value(mp.blocks[i].process.fs.curtailment))
    # print('pv size: ', value(mp.blocks[0].process.fs.pv.size))
    # print('battery power: ', value(mp.blocks[0].process.fs.battery.nameplate_power))
    # print('battery energy: ', value(mp.blocks[0].process.fs.battery.nameplate_energy))
    # print('total cost: ', value(mp.LCOW))

    # plt.show()