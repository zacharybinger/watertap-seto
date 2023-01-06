"""
This module contains a zero-order representation of a solar heat
operation.
"""

from idaes.core import declare_process_block_class

from watertap.core import build_pt, ZeroOrderBaseData
from watertap_contrib.seto.energy import solar_heat, solar_energy

__author__ = "Kurban Sitterley"


@declare_process_block_class("IPHTroughZO")
class IPHTroughZOData(ZeroOrderBaseData):
    """
    Zero-Order model for industrial process heat trough.
    """

    def build(self):
        super().build()

        self._tech_type = "iph_trough"

        build_pt(self)
        solar_heat(self)
        solar_energy(self)
