
# General python imports
import os
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
from pyomo.environ import Var, value, units as pyunits
__author__ = "Zachary Binger, Zhuoran Zhang, Mukta Hardikar"

absolute_path = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(absolute_path, os.pardir))
seto_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))

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
        # (t1.fs.pv.size, t2.fs.pv.size)
        ]

def unfix_dof(m):
    """
    This function unfixes a few degrees of freedom for optimization

    Args:
        m: object containing the integrated nuclear plant flowsheet

    Returns:
        None
    """
    # m.fs.battery.nameplate_energy.unfix()
    # m.fs.battery.nameplate_energy.fix(3000)
    m.fs.battery.nameplate_energy.unfix()
    m.fs.battery.nameplate_power.unfix()
    # m.fs.battery.initial_state_of_charge.unfix()
    # m.fs.battery.initial_energy_throughput.unfix()
    return

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
        if (day.month < 4) | day.month > 10:
            if (time < 12) | (time > 18):
                return electric_tiers['Tier 2']
            else:
                return electric_tiers['Tier 1']
        else:
            if (time < 12) | (time > 18):
                return electric_tiers['Tier 4']
            else:
                return electric_tiers['Tier 3']
    else:
        if (day.month < 4) | day.month > 10:
            return electric_tiers['Tier 2']
        else:
            return electric_tiers['Tier 4']

def create_multiperiod_pv_battery_model(
        n_time_points= 7*24,
        ro_capacity = 6000, # m3/day
        ro_elec_req = 944.3, # kW
        cost_battery_power = 75, # $/kW
        cost_battery_energy = 50, # $/kWh      
        elec_price = 0.15,
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
        surrogate = load_surrogate(surrogate_filename=parent_dir+'/net_metering/pysam_data/pv_Spring_Eq_surrogate_week.json')
        
    flowsheet_options={ t: { 
                            "pv_gen": max(0,eval_surrogate(surrogate, ro_elec_req, t%24, t//24)),
                            # "elec_price": elec_price[t],
                            "electricity_price": get_elec_tier(start_date+datetime.timedelta(days=t//24), t%24),
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

    # # initialize the beginning status of the system
    # mp.blocks[0].process.fs.battery.initial_state_of_charge.fix(0.8*mp.blocks[0].process.fs.battery.nameplate_energy)
    # for day in range(1,7):
    #     mp.blocks[day*24-1].process.fs.battery.initial_state_of_charge.fix(0.8*mp.blocks[day*24-1].process.fs.battery.nameplate_energy)
    
    # initialize the beginning status of the system
    mp.blocks[0].process.fs.battery.initial_state_of_charge.fix(0)
    # mp.blocks[0].process.fs.battery.initial_state_of_charge.fix(0)

    # for day in range(1,7):
    #     mp.blocks[day*24+5].process.fs.battery.initial_state_of_charge.fix(0)
        # mp.blocks[day*24-1].process.fs.battery.initial_state_of_charge.fix(1*mp.blocks[0].process.fs.battery.nameplate_energy)
    # Add battery cost function
    @mp.Expression(doc="battery cost")
    def battery_cost(b):
        return ( 0.168 * # capital recovery factor
            (cost_battery_power * b.blocks[0].process.fs.battery.nameplate_power
            +cost_battery_energy * b.blocks[0].process.fs.battery.nameplate_energy))
        
    # Add PV cost function
    @mp.Expression(doc="PV cost")
    def pv_cost(b):
        return (
            1040 * b.blocks[0].process.fs.pv_size * 0.168 # Annualized CAPEX
            + 9 * b.blocks[0].process.fs.pv_size)          # OPEX

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

def create_plot(mp, idx, elec_prices, norm=False):
    colors=['#235789', '#4A7C59', '#F1A208']
    n = 7*24
    titles = ['Summer','Winter','Spring', 'Fall']
    hour = [i for i in range(1,n+1)]
    battery_state = np.array([value(mp.blocks[i].process.fs.battery.state_of_charge[0]) for i in range(n)])
    pv_gen = np.array([value(mp.blocks[i].process.fs.elec_generation) for i in range(n)])
    pv_curtail = np.array([value(mp.blocks[i].process.fs.curtailment) for i in range(n)])
    electric_price = np.array([value(mp.blocks[i].process.fs.elec_price) for i in range(n)])
    ro_demand = np.array([value(mp.blocks[i].process.fs.elec_price) for i in range(n)])

    pv_to_ro = np.array([value(mp.blocks[i].process.fs.pv_to_ro) for i in range(n)])
    pv_to_battery = np.array([value(mp.blocks[i].process.fs.battery.elec_in[0]) for i in range(n)])
    battery_to_ro = np.array([value(mp.blocks[i].process.fs.battery.elec_out[0]) for i in range(n)])
    grid_to_ro = np.array([value(mp.blocks[i].process.fs.grid_to_ro) for i in range(n)])

    axes[idx].plot(hour, battery_state, 'r', label='Battery state (kWh)')
    axes[idx].plot(hour, pv_gen, 'k', label = 'PV generation (kWh)')
    axes[idx].plot(hour, pv_curtail, 'g', label = 'PV curtailment (kWh)')
    axes[idx].plot(hour, pv_to_battery, colors[2], label = 'PV to Battery (kWh)')
    axes[idx].vlines(x=[day*24 for day in range(7)],ymin=0,ymax=6000,linestyle='--',color='black')

    axes[idx].set_xlim([1,n])
    axes[idx].set_ylabel('  Energy (kWh)', loc='center', fontsize=16)
    axes[idx].set_xlabel('Operating Hours', fontsize=16)
    leg1 = axes[idx].legend(loc="lower left", frameon = False, bbox_to_anchor=(0, 1.0, 0.5, 1), ncols=4, mode="expand", borderaxespad=0.)
    axes[idx].set_title(titles[idx], loc='center', x=-0.07, y=0.5, rotation=90, fontweight='bold', ha='center', va='center', fontsize=16)
    axes[idx].tick_params(axis="x", labelsize=16)
    axes[idx].tick_params(axis="y", labelsize=16)
    ax3 = axes[idx].twinx()
    ax3.plot(hour, elec_prices,'--',label='Grid Price')
    ax3.set_ylabel('Grid Price ($/kWh)', ha='center', va='center', fontsize=16, labelpad=20)
    ax3.set_ylim([0,0.3])
    leg2 = ax3.legend(loc="lower left", frameon = False, bbox_to_anchor=(0.52, 1.0, 0.15, 1), ncols=1, mode="expand", borderaxespad=0.)
    ax3.tick_params(axis="y", labelsize=16)

    labels=["PV to RO", "Battery to RO", "Grid to RO", "PV to Battery"]
    norm_labels=["PV to RO", "Battery to RO", "PV to Battery", "Grid to RO"]
    
    frames = []
    for label in labels:
        frames.append(pd.DataFrame([hour, pv_to_ro, [label]*len(pv_to_ro)]).transpose())
    df2 = pd.concat(frames)
    df2.columns = ['Hour', 'State', 'Type']

    df = pd.DataFrame([hour, pv_to_ro, pv_to_battery, battery_to_ro, grid_to_ro]).transpose()
    df.columns = ["Hour", "PV to RO", "PV to Battery", "Battery to RO", "Grid to RO"]
    features = ["PV to RO", "PV to Battery", "Battery to RO", "Grid to RO"]
    df['Total'] = df["PV to RO"] + df["Battery to RO"] + df["Grid to RO"] + df["PV to Battery"]

    if norm == True:
        for feature in features:
            df[feature] = (df[feature]/df['Total'])
        axes2[idx].stackplot(hour, 100*df["PV to RO"],  100*df["Battery to RO"], 100*df["PV to Battery"], 100*df["Grid to RO"], baseline='zero', labels=norm_labels, alpha=1, ec='white')
        axes2[idx].set_ylabel('  Load %', loc='center', fontsize=16)
    else:
        axes2[idx].stackplot(hour, df["PV to RO"],  df["Battery to RO"], df["Grid to RO"], baseline='zero', colors=['#1f77b4','#ff7f0e','#d62728'], labels=labels, alpha=1, ec='white')
        axes2[idx].plot(hour, df["PV to Battery"], label="PV to Battery", color='#2ca02c', linewidth=2)
        axes2[idx].set_ylabel('  Power (kW)', loc='center', fontsize=16)

    axes2[idx].set_xlabel('Operation Hours', fontsize=16)
    leg3 = axes2[idx].legend(loc="lower left", frameon = False, bbox_to_anchor=(0, 1.0, 0.35, 1), ncols=4, mode="expand", borderaxespad=0.)
    if norm == True:
        axes2[idx].set_ylim([0,100])
        axes2[idx].yaxis.set_major_formatter(mtick.PercentFormatter()) 
        axes2[idx].vlines(x=[day*24 for day in range(7)],ymin=0,ymax=100,linestyle='--',color='black')
    axes2[idx].set_xlim([1,n])
    axes2[idx].set_title(titles[idx], loc='center', x=-0.07, y=0.5, rotation=90, fontweight='bold', ha='center', va='center', fontsize=16)
    axes2[idx].tick_params(axis="x", labelsize=16)
    axes2[idx].tick_params(axis="y", labelsize=16)
    ax4 = axes2[idx].twinx()
    ax4.plot(hour, elec_prices,linestyle='dotted', color='k',label='Grid Price')
    ax4.set_ylabel('Grid Price ($/kWh)', ha='center', va='center', fontsize=16, labelpad=20)
    ax4.set_ylim([0,0.3])
    leg4 = ax4.legend(loc="lower left", frameon = False, bbox_to_anchor=(0.37, 1.0, 0.15, 1),
        ncols=1, mode="expand", borderaxespad=0.)
    ax4.tick_params(axis="y", labelsize=16)

    
if __name__ == "__main__":
    rep_days, key_days = get_rep_weeks()
    fig,  axes= plt.subplots(4, figsize=(20,10))
    fig2,  axes2= plt.subplots(4, figsize=(20,10))
    surr = load_surrogate(surrogate_filename=join(parent_dir+'/net_metering/pysam_data/', "pv_"+'Winter Solstice'.replace(" ","_")+"_surrogate_week.json"))
    elec_prices = ([get_elec_tier(key_days[1]+datetime.timedelta(days=t//24), t%24) for t in range(7*24)])
    mp = create_multiperiod_pv_battery_model(surrogate = surr, start_date = key_days[1])
    results = solver.solve(mp)

    for idx, period in enumerate(['Summer Solstice','Winter Solstice','Spring Eq', 'Fall Eq']):
        surr = load_surrogate(surrogate_filename=join(parent_dir+'/net_metering/pysam_data/', "pv_"+period.replace(" ","_")+"_surrogate_week.json"))
        elec_prices = ([get_elec_tier(key_days[idx]+datetime.timedelta(days=t//24), t%24) for t in range(7*24)])
        mp = create_multiperiod_pv_battery_model(surrogate = surr, start_date = key_days[idx])
        results = solver.solve(mp)
        create_plot(mp, idx, elec_prices, norm=True)
        fig.tight_layout()
        fig2.tight_layout()
        print('pv size: ', value(mp.blocks[0].process.fs.pv_size))
        print('battery power: ', value(mp.blocks[0].process.fs.battery.nameplate_power))
        print('battery energy: ', value(mp.blocks[0].process.fs.battery.nameplate_energy))
        print('total cost: ', value(mp.LCOW))

    fig.savefig(absolute_path+'/plots/week_surrogate_battery_state.png', dpi=900)
    fig2.savefig(absolute_path+'/plots/week_surrogate_load2.png', dpi=900)
    plt.show()