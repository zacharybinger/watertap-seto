from pyomo.environ import ConcreteModel, NonNegativeReals, Var, ScalarVar, Set, Expression, SolverFactory, Reference, value, units as pyunits
from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In, ListOf

from idaes.core import FlowsheetBlock,UnitModelBlockData, declare_process_block_class
from idaes.core.util.exceptions import ConfigurationError
import idaes.logger as idaeslog

from watertap_contrib.seto.core import SolarEnergyBaseData

_log = idaeslog.getLogger(__name__)


@declare_process_block_class("EBC", doc="Performs energy balance between system and grid energy")
class GridSystemEnergyBalanceController(UnitModelBlockData):
    """
    Unit model to split a energy from a single inlet into multiple outlets based on split fractions
    """
    CONFIG= ConfigBlock()
    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False"
        )
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False"
        )
    )
    CONFIG.declare(
        "system_side_energy",
        ConfigValue(
            description="List of outlet names",
            doc="""A list containing names of outlets,
                **default** - None.
                **Valid values:** {
                **None** - use num_outlets argument,
                **list** - a list of names to use for outlets.}"""
        )
    )
    CONFIG.declare(
        "grid_side_sources",
        ConfigValue(
            domain=dict,
            description="List of outlet names",
            doc="""A list containing names of outlets,
                **default** - None.
                **Valid values:** {
                **None** - use num_outlets argument,
                **list** - a list of names to use for outlets.}"""
        )
    )
    CONFIG.declare(
        "costing",
        ConfigValue(
            description="Flowsheet costing block",
            doc="""Points to the flowsheet costing block"""
        )
    )

    def build(self):
        """

        """
        super().build()

        self.energy = Var(
            domain=NonNegativeReals,
            initialize=0.0,
            doc="Energy into control volume",
            units=pyunits.kW)
        
        self.system_side_energy = Var(
            domain=NonNegativeReals,
            initialize=self.config.system_side_energy,
            doc="Energy into control volume",
            units=pyunits.kW)
        
        self.grid_side_energy_buy = Var(
            domain=NonNegativeReals,
            initialize=0.0,
            doc="Energy purchased from grid",
            units=pyunits.kW)
        
        self.grid_side_energy_sell = Var(
            domain=NonNegativeReals,
            initialize=0.0,
            doc="Energy sold back to grid",
            units=pyunits.kW)
        
        self.add_grid_energy()
        
        @self.Constraint(doc="Buy Energy Balance")
        def eq_grid_side_buy(b):
            return b.grid_side_energy_buy == sum([getattr(self, flow + "_buy") for flow, costs in self.config.grid_side_sources.items()])
        
        @self.Constraint(doc="Sell Energy Balance")
        def eq_grid_side_sell(b):
            return b.grid_side_energy_sell == sum([getattr(self, flow + "_sell") for flow, costs in self.config.grid_side_sources.items()])

        @self.Constraint(doc="Energy Balance")
        def eq_energy_balance(b):
            return b.system_side_energy == b.grid_side_energy_buy - b.grid_side_energy_sell

        # [print(flow) for flow, costs in self.config.grid_side_sources.items()]
        sum([getattr(self, flow + "_buy") for flow, costs in self.config.grid_side_sources.items()])
        sum([getattr(self, flow + "_sell") for flow, costs in self.config.grid_side_sources.items()])
    
    def add_grid_energy(self):
        """
        """
        config = self.config

        for flow, prices in config.grid_side_sources.items():
            grid_flow_buy_cost = Var(
                domain=NonNegativeReals,
                initialize=prices['buy_price'],
                doc="Energy at outlet {}".format(flow),
                units=pyunits.kW)
            setattr(self, flow+"_buy", grid_flow_buy_cost)
            config.costing.add_defined_flow(flow+"_buy", grid_flow_buy_cost)
            config.costing.cost_flow(flow+"_buy", f"{flow}_buy")

            grid_flow_sell_cost = Var(
                domain=NonNegativeReals,
                initialize=prices['sell_price'],
                doc="Energy at outlet {}".format(flow),
                units=pyunits.kW)
            setattr(self, flow+"_sell", grid_flow_sell_cost)
            config.costing.add_defined_flow(flow+"_sell", grid_flow_sell_cost)
            config.costing.cost_flow(flow+"_sell", f"{flow}_sell")
            